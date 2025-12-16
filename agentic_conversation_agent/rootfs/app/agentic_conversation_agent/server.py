from __future__ import annotations

import logging
import time
from typing import Any

from pathlib import Path

from aiohttp import web

from .agent import Agent
from .config import load_config
from .embeddings import build_embeddings_provider
from .llm import build_llm_client
from .logging_utils import configure_logging
from .memory import BufferStore, ConversationContext, MemoryManager
from .tools import load_tools_from_dir
from .vector_store import VectorStoreSQLite


_LOGGER = logging.getLogger(__name__)


def _dir_mtime(path: str) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    latest = 0.0
    for child in p.rglob("*"):
        try:
            if child.is_file():
                latest = max(latest, child.stat().st_mtime)
        except Exception:  # noqa: BLE001
            continue
    return latest


def _require_api_key(app: web.Application, request: web.Request) -> None:
    api_key = app["cfg"].api_key
    if not api_key:
        return
    provided = request.headers.get("X-API-Key", "")
    if provided != api_key:
        raise web.HTTPUnauthorized(text="Missing/invalid X-API-Key")


async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def reload_tools(request: web.Request) -> web.Response:
    _require_api_key(request.app, request)
    agent: Agent = request.app["agent"]
    cfg = request.app["cfg"]

    tools = load_tools_from_dir(cfg.tools.dir)
    await agent.reload_tools(tools=tools)

    # Re-index tools embeddings in vector store
    await _index_tools(request.app, tools)
    return web.json_response({"ok": True, "tools": len(tools)})


async def converse(request: web.Request) -> web.Response:
    _require_api_key(request.app, request)
    agent: Agent = request.app["agent"]

    # Auto-reload tools if the user dropped/edited files.
    cfg = request.app["cfg"]
    current_mtime = _dir_mtime(cfg.tools.dir)
    if current_mtime > float(request.app.get("tools_mtime", 0.0)):
        _LOGGER.info("Tools directory changed; reloading")
        tools = load_tools_from_dir(cfg.tools.dir)
        await agent.reload_tools(tools=tools)
        await _index_tools(request.app, tools)
        request.app["tools_mtime"] = current_mtime

    payload = await request.json()
    text = str(payload.get("text", "")).strip()
    if not text:
        raise web.HTTPBadRequest(text="Missing text")

    ctx = ConversationContext(
        speaker=(payload.get("speaker") or None),
        device_id=(payload.get("device_id") or None),
        user_id=(payload.get("user_id") or None),
        conversation_id=(payload.get("conversation_id") or None),
        language=(payload.get("language") or None),
    )

    try:
        result = await agent.converse(text=text, ctx=ctx)
    except Exception as err:  # noqa: BLE001
        _LOGGER.exception("Converse failed")
        return web.json_response(
            {
                "text": f"Sorry — I hit an error: {err}",
                "continue_conversation": False,
                "conversation_id": ctx.conversation_id,
            },
            status=200,
        )

    return web.json_response(
        {
            "text": result.text,
            "continue_conversation": result.continue_conversation,
            "conversation_id": result.conversation_id,
        }
    )


async def _index_tools(app: web.Application, tools: list[Any]) -> None:
    cfg = app["cfg"]
    store: VectorStoreSQLite = app["store"]
    embeddings = app["embeddings"]

    # Drop prior tool vectors and rebuild.
    store.delete_prefix(kind="tool", prefix="tool:")

    for t in tools:
        text = f"{t.name}\n{t.description}"
        emb = await embeddings.embed(text)
        store.upsert(
            item_id=f"tool:{t.name}",
            kind="tool",
            text=text,
            embedding=emb,
            metadata={"tool_name": t.name},
            expires_at=None,
        )


async def init_app() -> web.Application:
    cfg = load_config()
    configure_logging(cfg.log_level)

    store = VectorStoreSQLite(cfg.db_path)

    llm = build_llm_client(
        provider=cfg.llm.provider,
        model=cfg.llm.model,
        api_key=cfg.llm.api_key,
        base_url=cfg.llm.base_url,
    )

    summarizer_llm = None
    if cfg.memory.summarization_model and cfg.memory.summarization_model != cfg.llm.model:
        summarizer_llm = build_llm_client(
            provider=cfg.llm.provider,
            model=cfg.memory.summarization_model,
            api_key=cfg.llm.api_key,
            base_url=cfg.llm.base_url,
        )

    embeddings = build_embeddings_provider(
        provider=cfg.embeddings.provider,
        model=cfg.embeddings.model,
        api_key=cfg.embeddings.api_key,
        base_url=cfg.embeddings.base_url,
    )

    tools = load_tools_from_dir(cfg.tools.dir)

    memory = MemoryManager(store=store)
    buffer = BufferStore(db_path=cfg.db_path)

    agent = Agent(
        config=cfg,
        llm=llm,
        summarizer_llm=summarizer_llm,
        embeddings=embeddings,
        store=store,
        tools=tools,
        memory=memory,
        buffer=buffer,
    )

    app = web.Application()
    app["cfg"] = cfg
    app["agent"] = agent
    app["store"] = store
    app["embeddings"] = embeddings
    app["tools_mtime"] = _dir_mtime(cfg.tools.dir)

    app.add_routes(
        [
            web.get("/health", health),
            web.post("/reload_tools", reload_tools),
            web.post("/converse", converse),
        ]
    )

    await _index_tools(app, tools)
    store.prune_expired()

    return app


def main() -> None:
    loop_time = time.time()
    cfg = load_config()
    configure_logging(cfg.log_level)
    _LOGGER.info("Starting server on %s:%s", cfg.listen_host, cfg.listen_port)
    web.run_app(init_app(), host=cfg.listen_host, port=cfg.listen_port)
    _LOGGER.debug("Server stopped after %s", time.time() - loop_time)


if __name__ == "__main__":
    main()
