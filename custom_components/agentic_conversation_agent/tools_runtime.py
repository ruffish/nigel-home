from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import yaml

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    type: str  # ha_service | http | python | builtin
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

        tools.append(ToolSpec(name=name, description=desc, type=tool_type, args_schema=args_schema, raw=data))

    unique: dict[str, ToolSpec] = {}
    for t in tools:
        unique[t.name] = t
    return list(unique.values())


def builtin_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="ha_list_services",
            description=(
                "List available Home Assistant services and their fields. "
                "Use when you are not sure which domain.service exists or what data fields it accepts."
            ),
            type="builtin",
            args_schema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "Optional domain filter (e.g. light)"},
                    "query": {"type": "string", "description": "Optional substring filter on domain.service"},
                    "limit": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            raw={},
        ),
        ToolSpec(
            name="ha_list_entities",
            description=(
                "List entities (entity_id + name + state). Use to discover entity_ids to control."
            ),
            type="builtin",
            args_schema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "Optional domain filter (e.g. light)"},
                    "query": {"type": "string", "description": "Optional substring filter on name/entity_id"},
                    "limit": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            raw={},
        ),
        ToolSpec(
            name="ha_call_service",
            description=(
                "Call a Home Assistant service. Use for device control, timers (timer.* entities), scripts, etc."
            ),
            type="builtin",
            args_schema={
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "domain.service"},
                    "data": {"type": "object", "description": "Service data payload"},
                    "target": {"type": "object", "description": "Optional service target (entity_id, device_id, area_id, label_id)"},
                },
                "required": ["service"],
            },
            raw={},
        ),
        ToolSpec(
            name="ha_get_state",
            description="Get the current state + attributes for an entity_id.",
            type="builtin",
            args_schema={
                "type": "object",
                "properties": {"entity_id": {"type": "string"}},
                "required": ["entity_id"],
            },
            raw={},
        ),
        ToolSpec(
            name="ha_find_entities",
            description="Search entities by partial name/domain to find the right entity_id.",
            type="builtin",
            args_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "domain": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
            raw={},
        ),
        ToolSpec(
            name="local_timer",
            description=(
                "Local per-conversation timer when HA device-scoped timers are unavailable. "
                "Actions: start/cancel/remaining."
            ),
            type="builtin",
            args_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["start", "cancel", "remaining"]},
                    "seconds": {"type": "integer"},
                },
                "required": ["action"],
            },
            raw={},
        ),
        ToolSpec(
            name="memory_write",
            description=(
                "Save a durable memory about the user (preferences, names, stable facts). "
                "Only store what the user explicitly asked you to remember."
            ),
            type="builtin",
            args_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "expires_days": {"type": "integer"},
                },
                "required": ["text"],
            },
            raw={},
        ),
    ]


def tool_prompt_block(tools: list[ToolSpec]) -> str:
    lines: list[str] = []
    for t in sorted(tools, key=lambda x: x.name):
        lines.append(f"- {t.name}: {t.description}")
        lines.append(f"  type: {t.type}")
        lines.append(f"  args_schema: {json.dumps(t.args_schema, ensure_ascii=False)}")
    return "\n".join(lines)


async def execute_tool(
    *,
    hass: HomeAssistant,
    tool: ToolSpec,
    args: dict[str, Any],
    allow_python: bool,
    session: aiohttp.ClientSession,
    local_timer_handler: Any,
    memory_write_handler: Any,
) -> dict[str, Any]:
    # Built-in tools
    if tool.name == "ha_list_services":
        domain_filter = str(args.get("domain") or "").strip().lower() or None
        query = str(args.get("query") or "").strip().lower() or None
        limit = int(args.get("limit") or 200)
        limit = max(1, min(limit, 500))

        services = hass.services.async_services()
        out: list[dict[str, Any]] = []
        for domain, svc_map in services.items():
            if domain_filter and domain.lower() != domain_filter:
                continue
            for svc, info in svc_map.items():
                full = f"{domain}.{svc}".lower()
                if query and query not in full:
                    continue

                entry: dict[str, Any] = {"service": f"{domain}.{svc}"}
                if isinstance(info, dict):
                    desc = info.get("description")
                    if isinstance(desc, str) and desc:
                        entry["description"] = desc
                    fields = info.get("fields")
                    if isinstance(fields, dict):
                        # Only include field names + short descriptions; keep payload small.
                        entry["fields"] = {
                            str(k): (
                                (str(v.get("description")) if isinstance(v, dict) and v.get("description") else "")
                            )
                            for k, v in list(fields.items())[:50]
                        }

                out.append(entry)
                if len(out) >= limit:
                    return {"services": out, "truncated": True}

        return {"services": out, "truncated": False}

    if tool.name == "ha_list_entities":
        domain_filter = str(args.get("domain") or "").strip().lower() or None
        query = str(args.get("query") or "").strip().lower() or None
        limit = int(args.get("limit") or 100)
        limit = max(1, min(limit, 500))

        entities: list[dict[str, Any]] = []
        for st in hass.states.async_all():
            if domain_filter and st.domain != domain_filter:
                continue
            name = (st.name or "")
            if query:
                q = query
                if q not in st.entity_id.lower() and q not in name.lower():
                    continue
            entities.append(
                {
                    "entity_id": st.entity_id,
                    "name": name,
                    "domain": st.domain,
                    "state": st.state,
                }
            )
            if len(entities) >= limit:
                return {"entities": entities, "truncated": True}

        return {"entities": entities, "truncated": False}

    if tool.name == "ha_call_service":
        svc = str(args.get("service", ""))
        data = args.get("data") if isinstance(args.get("data"), dict) else {}
        target = args.get("target") if isinstance(args.get("target"), dict) else None
        if "." not in svc:
            raise ValueError("service must be domain.service")
        domain, service = svc.split(".", 1)
        await hass.services.async_call(domain, service, data, blocking=True, target=target)
        return {"ok": True, "service": svc}

    if tool.name == "ha_get_state":
        entity_id = str(args.get("entity_id", "")).strip()
        st = hass.states.get(entity_id)
        if st is None:
            return {"found": False, "entity_id": entity_id}
        return {
            "found": True,
            "entity_id": entity_id,
            "state": st.state,
            "attributes": dict(st.attributes),
        }

    if tool.name == "ha_find_entities":
        query = str(args.get("query", "")).strip().lower()
        domain = str(args.get("domain", "")).strip().lower() or None
        limit = int(args.get("limit", 10) or 10)
        matches: list[dict[str, Any]] = []
        for st in hass.states.async_all():
            if domain and st.domain != domain:
                continue
            name = (st.name or "").lower()
            if query in name or query in st.entity_id.lower():
                matches.append({"entity_id": st.entity_id, "name": st.name, "domain": st.domain})
            if len(matches) >= limit:
                break
        return {"matches": matches}

    if tool.name == "local_timer":
        return await local_timer_handler(args)

    if tool.name == "memory_write":
        return await memory_write_handler(args)

    # User-defined tools
    if tool.type == "ha_service":
        svc = str(tool.raw.get("service", ""))
        if "." not in svc:
            raise ValueError(f"Tool '{tool.name}' has invalid service: {svc}")
        domain, service = svc.split(".", 1)
        base_data = tool.raw.get("data") if isinstance(tool.raw.get("data"), dict) else {}
        base_target = tool.raw.get("target") if isinstance(tool.raw.get("target"), dict) else None

        payload: dict[str, Any] = {}
        payload.update(base_data)
        payload.update(args or {})

        call_target = payload.pop("target", None)
        if not isinstance(call_target, dict):
            call_target = None
        if call_target is None:
            call_target = base_target

        await hass.services.async_call(domain, service, payload, blocking=True, target=call_target)
        return {"ok": True, "service": svc}

    if tool.type == "http":
        method = str(tool.raw.get("method", "GET")).upper()
        url = str(tool.raw.get("url", ""))
        if not url:
            raise ValueError(f"Tool '{tool.name}' missing url")

        headers = tool.raw.get("headers") if isinstance(tool.raw.get("headers"), dict) else {}
        query = tool.raw.get("query") if isinstance(tool.raw.get("query"), dict) else {}
        json_body = tool.raw.get("json") if isinstance(tool.raw.get("json"), dict) else None
        timeout_s = float(tool.raw.get("timeout_s", 20))

        if json_body is not None:
            merged = {}
            merged.update(json_body)
            merged.update(args or {})
            json_body = merged

        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with session.request(method, url, headers=headers, params=query, json=json_body, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "application/json" in content_type:
                data = await resp.json()
            else:
                data = await resp.text()
            return {"status": resp.status, "data": data}

    if tool.type == "python":
        if not allow_python:
            raise ValueError("Python tools are disabled in integration options")

        script_path = str(tool.raw.get("script_path", ""))
        if not script_path:
            raise ValueError(f"Tool '{tool.name}' missing script_path")

        # Safety: only allow scripts under HA config directory by default.
        config_dir = hass.config.path("")
        resolved = str(Path(script_path).expanduser().resolve())
        if not resolved.startswith(str(Path(config_dir).resolve())):
            raise ValueError("Python tool script_path must be under the Home Assistant config directory")

        async def _run() -> dict[str, Any]:
            proc = await asyncio.create_subprocess_exec(
                "python3" if os.name != "nt" else "python",
                resolved,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate(json.dumps(args or {}, ensure_ascii=False).encode("utf-8"))
            out = stdout.decode("utf-8", errors="replace").strip()
            err = stderr.decode("utf-8", errors="replace").strip()
            result: dict[str, Any] = {"returncode": int(proc.returncode or 0)}
            if out:
                try:
                    result["stdout_json"] = json.loads(out)
                except Exception:  # noqa: BLE001
                    result["stdout"] = out
            if err:
                result["stderr"] = err
            return result

        return await _run()

    raise ValueError(f"Unknown tool type: {tool.type}")
