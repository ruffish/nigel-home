"""Simple HTTP wrapper around GitHub Copilot CLI.

POST /copilot
Body: {"prompt": "..."}
Response: {"ok": true, "response": "..."}

Health: GET /health -> {"ok": true}
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import Any

from aiohttp import web

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("copilot_server")


async def _run_copilot(prompt: str) -> str:
    loop = asyncio.get_running_loop()

    def _do() -> str:
        result = subprocess.run(
            ["copilot", "-i", prompt],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(f"Copilot CLI failed ({result.returncode}): {stderr[:300]}")
        return (result.stdout or "").strip()

    return await loop.run_in_executor(None, _do)


async def copilot_handler(request: web.Request) -> web.Response:
    try:
        data: Any = await request.json()
        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            return web.json_response({"ok": False, "error": "prompt required"}, status=400)

        text = await _run_copilot(prompt)
        return web.json_response({"ok": True, "response": text})
    except asyncio.TimeoutError:
        return web.json_response({"ok": False, "error": "copilot timeout"}, status=504)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("Copilot handler error")
        return web.json_response({"ok": False, "error": str(exc)}, status=500)


async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


def build_app() -> web.Application:
    app = web.Application()
    app.add_routes([
        web.get("/health", health),
        web.post("/copilot", copilot_handler),
    ])
    return app


def main() -> None:
    _LOGGER.info("Starting Copilot HTTP server on 0.0.0.0:8100")
    web.run_app(build_app(), host="0.0.0.0", port=8100)


if __name__ == "__main__":
    main()
