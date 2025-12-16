from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import yaml
from jsonschema import Draft202012Validator

from .utils import UserVisibleError


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    type: str  # ha_service | http | python
    args_schema: dict[str, Any]
    raw: dict[str, Any]


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def load_tools_from_dir(tools_dir: str) -> list[ToolSpec]:
    base = Path(tools_dir)
    if not base.exists():
        return []

    tools: list[ToolSpec] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
            continue
        try:
            data = _load_yaml_or_json(path)
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning("Failed loading tool file %s: %s", path, err)
            continue

        name = str(data.get("name", "")).strip()
        desc = str(data.get("description", "")).strip()
        tool_type = str(data.get("type", "")).strip()
        if not name or not desc or not tool_type:
            _LOGGER.warning("Skipping tool file %s (missing name/description/type)", path)
            continue

        args_schema = data.get("args_schema")
        if not isinstance(args_schema, dict):
            args_schema = {"type": "object", "properties": {}, "additionalProperties": True}

        tools.append(
            ToolSpec(
                name=name,
                description=desc,
                type=tool_type,
                args_schema=args_schema,
                raw=data,
            )
        )

    # Unique by name (last wins)
    unique: dict[str, ToolSpec] = {}
    for t in tools:
        unique[t.name] = t
    return list(unique.values())


def tool_prompt_block(tools: list[ToolSpec]) -> str:
    lines: list[str] = []
    for t in sorted(tools, key=lambda x: x.name):
        lines.append(f"- {t.name}: {t.description}")
        lines.append(f"  type: {t.type}")
        lines.append(f"  args_schema: {json.dumps(t.args_schema, ensure_ascii=False)}")
    return "\n".join(lines)


def validate_tool_args(tool: ToolSpec, args: dict[str, Any]) -> None:
    try:
        Draft202012Validator(tool.args_schema).validate(args)
    except Exception as err:  # noqa: BLE001
        raise UserVisibleError(f"Invalid args for tool '{tool.name}': {err}") from err


async def execute_tool(
    *,
    tool: ToolSpec,
    args: dict[str, Any],
    allow_python: bool,
    session: aiohttp.ClientSession,
) -> dict[str, Any]:
    validate_tool_args(tool, args)

    if tool.type == "ha_service":
        return await _exec_ha_service(tool=tool, args=args, session=session)
    if tool.type == "http":
        return await _exec_http(tool=tool, args=args, session=session)
    if tool.type == "python":
        if not allow_python:
            raise UserVisibleError("Python tools are disabled in add-on config")
        return await _exec_python(tool=tool, args=args)

    raise UserVisibleError(f"Unknown tool type: {tool.type}")


async def _exec_ha_service(*, tool: ToolSpec, args: dict[str, Any], session: aiohttp.ClientSession) -> dict[str, Any]:
    svc = str(tool.raw.get("service", ""))
    if "." not in svc:
        raise UserVisibleError(f"Tool '{tool.name}' has invalid service: {svc}")
    domain, service = svc.split(".", 1)

    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        raise UserVisibleError("SUPERVISOR_TOKEN not available; is this running as a HA add-on?")

    base_data = tool.raw.get("data") if isinstance(tool.raw.get("data"), dict) else {}
    target = tool.raw.get("target") if isinstance(tool.raw.get("target"), dict) else {}

    payload = {}
    payload.update(base_data)
    payload.update(args or {})
    if target:
        # REST service calls commonly accept these keys at the top level.
        for key in ("entity_id", "device_id", "area_id", "label_id", "floor_id"):
            if key in target:
                payload[key] = target[key]

    url = f"http://supervisor/core/api/services/{domain}/{service}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    async with session.post(url, headers=headers, json=payload) as resp:
        text = await resp.text()
        if resp.status >= 300:
            raise UserVisibleError(f"HA service call failed ({resp.status}): {text}")

    return {"ok": True, "service": svc}


async def _exec_http(*, tool: ToolSpec, args: dict[str, Any], session: aiohttp.ClientSession) -> dict[str, Any]:
    method = str(tool.raw.get("method", "GET")).upper()
    url = str(tool.raw.get("url", ""))
    if not url:
        raise UserVisibleError(f"Tool '{tool.name}' missing url")

    headers = tool.raw.get("headers") if isinstance(tool.raw.get("headers"), dict) else {}
    query = tool.raw.get("query") if isinstance(tool.raw.get("query"), dict) else {}
    json_body = tool.raw.get("json") if isinstance(tool.raw.get("json"), dict) else None
    timeout_s = float(tool.raw.get("timeout_s", 20))

    # args may be merged into json body if present
    if json_body is not None:
        merged = {}
        merged.update(json_body)
        merged.update(args or {})
        json_body = merged

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with session.request(
        method,
        url,
        headers=headers,
        params=query,
        json=json_body,
        timeout=timeout,
    ) as resp:
        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type:
            data = await resp.json()
        else:
            data = await resp.text()
        return {"status": resp.status, "data": data}


async def _exec_python(*, tool: ToolSpec, args: dict[str, Any]) -> dict[str, Any]:
    script_path = str(tool.raw.get("script_path", ""))
    if not script_path:
        raise UserVisibleError(f"Tool '{tool.name}' missing script_path")

    # Safety: only allow scripts under /data
    if not Path(script_path).resolve().as_posix().startswith("/data/"):
        raise UserVisibleError("Python tool script_path must be under /data/")

    def _run() -> dict[str, Any]:
        proc = subprocess.run(
            ["python3", script_path],
            input=json.dumps(args, ensure_ascii=False).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        out = proc.stdout.decode("utf-8", errors="replace").strip()
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        result: dict[str, Any] = {"returncode": proc.returncode}
        if out:
            try:
                result["stdout_json"] = json.loads(out)
            except Exception:  # noqa: BLE001
                result["stdout"] = out
        if err:
            result["stderr"] = err
        return result

    return await asyncio.to_thread(_run)
