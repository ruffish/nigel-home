from __future__ import annotations

import logging
import json
import re
import time
import asyncio
from typing import TYPE_CHECKING, Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers import intent as ha_intent

try:
    # Home Assistant 2024+ (and current) location
    from homeassistant.helpers.intent import IntentResponse
except ImportError:  # pragma: no cover
    # Older HA versions (best-effort fallback)
    from homeassistant.components.intent import IntentResponse  # type: ignore[attr-defined]
from homeassistant.components.conversation import (
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
)

if TYPE_CHECKING:
    from homeassistant.components.conversation.chat_log import ChatLog

from .const import (
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_LLM_PROVIDER,
    CONF_EMBEDDINGS_API_KEY,
    CONF_EMBEDDINGS_BASE_URL,
    CONF_EMBEDDINGS_MODEL,
    CONF_EMBEDDINGS_PROVIDER,
    CONF_TOOLS_DIR,
    CONF_ALLOW_PYTHON,
    CONF_AGENT_MAX_STEPS,
    CONF_TOOL_TOP_K,
    CONF_MEMORY_TOP_K,
    CONF_MEMORY_EXPIRES_DAYS,
    CONF_BUFFER_MAX_TURNS,
    DEFAULT_ALLOW_PYTHON,
    DEFAULT_AGENT_MAX_STEPS,
    DEFAULT_BUFFER_MAX_TURNS,
    DEFAULT_MEMORY_EXPIRES_DAYS,
    DEFAULT_MEMORY_TOP_K,
    DEFAULT_TOOL_TOP_K,
    CONF_INDEX_HA_SERVICES,
    DEFAULT_INDEX_HA_SERVICES,
    DOMAIN,
)
from .embeddings import build_embeddings_provider
from .llm import LLMClient, LLMConfig
from .memory_runtime import BufferStore, ConversationContext, MemoryManager, expiry_timestamp
from .tools_runtime import ToolSpec, builtin_tools, execute_tool, load_tools_from_dir, tool_prompt_block
from .vector_store import VectorStoreSQLite

_LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful Home Assistant voice assistant named Nigel.
You help users control their smart home and answer questions.
Be concise and friendly. If you don't know something, say so.
When the user asks to control devices, explain what you would do (you don't have direct device control yet)."""


AGENT_PROMPT = """You are Nigel, a Home Assistant voice assistant.

You can call tools to control the home and fetch state.

Important:
- NEVER delegate to Home Assistant's DefaultAgent or built-in conversation agent.
- All actions must be performed by calling tools.

Tool calling rules:
- Output MUST be STRICT JSON only (no markdown, no extra text).
- Choose exactly one action per step.

Valid outputs:

1) Call a tool:
{ "action": "tool", "tool": "<tool_name>", "args": { ... } }

2) Finish:
{ "action": "final", "final_text": "..." }

Guidance:
- If you are unsure what service exists or which fields it needs, call ha_list_services.
- If you are unsure what entity_id to control, call ha_list_entities or ha_find_entities.
- For device control use ha_call_service or a matching ha_service tool.
- To find an entity_id, use ha_find_entities; then ha_get_state.
- For timers when the client device_id is missing, use local_timer.
- Only call memory_write when the user explicitly asked you to remember something.
"""


ROUTER_PROMPT = """(Deprecated)"""


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first JSON object from model output."""
    if not text:
        return None

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    obj = json.loads(candidate)
                except Exception:
                    return None
                return obj if isinstance(obj, dict) else None

    return None


_DURATION_PART_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)\b",
    re.IGNORECASE,
)


def _parse_duration_seconds(text: str) -> int | None:
    """Parse durations like '5 minutes', '1h 30m', '90 seconds' into seconds."""
    if not text:
        return None

    total = 0.0
    found = False
    for match in _DURATION_PART_RE.finditer(text):
        found = True
        value = float(match.group("value"))
        unit = match.group("unit").lower()
        if unit.startswith("h"):
            total += value * 3600
        elif unit.startswith("m"):
            total += value * 60
        else:
            total += value

    if not found:
        return None

    seconds = int(round(total))
    return seconds if seconds > 0 else None


def _looks_like_timer_request(text: str) -> bool:
    t = (text or "").lower()
    return "timer" in t


def _timer_intent_kind(text: str) -> str | None:
    """Heuristic classification for simple timer operations."""
    t = (text or "").lower().strip()
    if not t:
        return None

    if any(k in t for k in ("cancel", "stop", "delete", "remove")) and "timer" in t:
        return "cancel"

    if "timer" in t and any(k in t for k in ("how long", "time left", "remaining", "left")):
        return "remaining"

    if "timer" in t and any(k in t for k in ("start", "set", "begin", "create")):
        return "start"

    return None


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    async_add_entities([AgenticConversationEntity(hass, entry)])


class AgenticConversationEntity(ConversationEntity):
    _attr_supported_features = ConversationEntityFeature.CONTROL
    _attr_has_entity_name = True

    @property
    def supported_languages(self) -> list[str] | str:
        """Return supported languages."""
        return "*"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._entry = entry
        cfg = {**entry.data, **entry.options}
        self._attr_name = cfg.get("name") or "Agentic Conversation Agent"
        self._attr_unique_id = f"{entry.entry_id}"

        # LLM configuration
        self._llm_config = LLMConfig(
            provider=str(cfg.get(CONF_LLM_PROVIDER, "google")),
            model=str(cfg.get(CONF_LLM_MODEL, "gemini-1.5-flash")),
            api_key=str(cfg.get(CONF_LLM_API_KEY, "")),
            base_url=cfg.get(CONF_LLM_BASE_URL) or None,
        )
        self._llm = LLMClient(self._llm_config)

        # Agentic configuration
        self._allow_python = bool(cfg.get(CONF_ALLOW_PYTHON, DEFAULT_ALLOW_PYTHON))
        self._agent_max_steps = int(cfg.get(CONF_AGENT_MAX_STEPS, DEFAULT_AGENT_MAX_STEPS))
        self._tool_top_k = int(cfg.get(CONF_TOOL_TOP_K, DEFAULT_TOOL_TOP_K))
        self._memory_top_k = int(cfg.get(CONF_MEMORY_TOP_K, DEFAULT_MEMORY_TOP_K))
        self._memory_expires_days = int(cfg.get(CONF_MEMORY_EXPIRES_DAYS, DEFAULT_MEMORY_EXPIRES_DAYS))
        self._buffer_max_turns = int(cfg.get(CONF_BUFFER_MAX_TURNS, DEFAULT_BUFFER_MAX_TURNS))
        self._tools_dir = str(cfg.get(CONF_TOOLS_DIR, "") or "").strip()
        self._index_ha_services = bool(cfg.get(CONF_INDEX_HA_SERVICES, DEFAULT_INDEX_HA_SERVICES))

        try:
            self._embeddings = build_embeddings_provider(
                provider=str(cfg.get(CONF_EMBEDDINGS_PROVIDER, "simple")),
                model=(str(cfg.get(CONF_EMBEDDINGS_MODEL, "")) or None),
                api_key=(str(cfg.get(CONF_EMBEDDINGS_API_KEY, "")) or None),
                base_url=(str(cfg.get(CONF_EMBEDDINGS_BASE_URL, "")) or None),
            )
        except Exception:  # noqa: BLE001
            from .embeddings import SimpleEmbeddings

            _LOGGER.exception("Embeddings provider misconfigured; falling back to local simple embeddings")
            self._embeddings = SimpleEmbeddings()

        # Persistent stores (SQLite) for tool vectors + memory + buffer.
        db_path = hass.config.path(".storage", f"{DOMAIN}.sqlite")
        self._store = VectorStoreSQLite(hass=hass, db_path=db_path)
        self._store_ready = False
        self._memory = MemoryManager(store=self._store)
        self._buffer = BufferStore(store=self._store)

        self._tool_specs_by_name: dict[str, ToolSpec] = {}
        self._tools_indexed = False

        # Simple in-integration timers (for clients that don't provide a device_id).
        # Keyed by a stable timer key (conversation_id/user_id/device_id).
        self._timers: dict[str, dict[str, float]] = {}

        self._last_prune_ts = 0.0

    async def _async_ensure_store(self) -> None:
        if self._store_ready:
            return
        await self._store.async_setup()
        self._store_ready = True

    async def _async_prune_if_needed(self) -> None:
        now = time.time()
        if now - self._last_prune_ts < 3600:
            return
        self._last_prune_ts = now
        try:
            await self._memory.prune()
        except Exception:  # noqa: BLE001
            _LOGGER.debug("Memory prune failed", exc_info=True)

    async def _async_index_tools(self) -> None:
        if self._tools_indexed:
            return

        await self._async_ensure_store()

        # Load tools (built-in + optional user-defined)
        tools: list[ToolSpec] = []
        tools.extend(builtin_tools())
        if self._tools_dir:
            tools.extend(load_tools_from_dir(self._tools_dir))

        # Index Home Assistant built-in services as ha_service tools for RAG.
        # This makes "default actions" (what DefaultAgent would do) directly retrievable and callable by the LLM.
        if self._index_ha_services:
            try:
                tools.extend(self._build_service_tools())
            except Exception:  # noqa: BLE001
                _LOGGER.debug("Failed building HA service tools", exc_info=True)

        unique: dict[str, ToolSpec] = {}
        for t in tools:
            unique[t.name] = t
        tools = list(unique.values())

        # Rebuild tool index prefix each startup.
        await self._store.delete_prefix(kind="tool", prefix="tool:")

        for t in tools:
            text = f"{t.name}: {t.description}\nargs_schema: {json.dumps(t.args_schema, ensure_ascii=False)}"
            try:
                emb = await self._embeddings.embed(text)
            except Exception:  # noqa: BLE001
                # If embeddings provider is misconfigured, fall back to local simple embeddings.
                from .embeddings import SimpleEmbeddings

                emb = await SimpleEmbeddings().embed(text)

            await self._store.upsert(
                item_id=f"tool:{t.name}",
                kind="tool",
                text=text,
                embedding=emb,
                metadata={"name": t.name, "type": t.type},
                expires_at=None,
            )

        self._tool_specs_by_name = {t.name: t for t in tools}
        self._tools_indexed = True

    def _build_service_tools(self) -> list[ToolSpec]:
        services = self.hass.services.async_services()
        out: list[ToolSpec] = []
        for domain, svc_map in services.items():
            for svc, info in svc_map.items():
                full = f"{domain}.{svc}"

                desc = ""
                fields: dict[str, Any] = {}
                if isinstance(info, dict):
                    d = info.get("description")
                    if isinstance(d, str):
                        desc = d.strip()
                    f = info.get("fields")
                    if isinstance(f, dict):
                        fields = f

                props: dict[str, Any] = {}
                for field_name, field_info in list(fields.items())[:100]:
                    if not isinstance(field_name, str):
                        continue
                    field_desc = ""
                    if isinstance(field_info, dict):
                        fd = field_info.get("description")
                        if isinstance(fd, str):
                            field_desc = fd
                    props[field_name] = {"type": "string", "description": field_desc}

                # Allow passing 'target' separately (entity_id/device_id/area_id/label_id).
                props.setdefault(
                    "target",
                    {
                        "type": "object",
                        "description": "Optional service target (entity_id, device_id, area_id, label_id)",
                        "additionalProperties": True,
                    },
                )

                args_schema = {
                    "type": "object",
                    "properties": props,
                    "additionalProperties": True,
                }

                out.append(
                    ToolSpec(
                        name=full,
                        description=desc or f"Call Home Assistant service {full}.",
                        type="ha_service",
                        args_schema=args_schema,
                        raw={"service": full},
                    )
                )

        return out

    async def _async_timer_finished(self, *, conversation_id: str, ends_at: float) -> None:
        delay = max(0.0, ends_at - time.time())
        await asyncio.sleep(delay)

        timer = self._timers.get(conversation_id)
        if not timer or timer.get("ends_at") != ends_at:
            return

        # Mark as finished but keep record so "time left" can answer.
        timer["finished"] = 1.0
        try:
            from homeassistant.components import persistent_notification

            persistent_notification.async_create(
                self.hass,
                "Timer finished.",
                title="Assist Timer",
                notification_id=f"{DOMAIN}_{conversation_id}_timer",
            )
        except Exception:  # noqa: BLE001
            _LOGGER.debug("Unable to create persistent notification for timer", exc_info=True)

    def _timer_key(self, *, ctx: ConversationContext, conversation_id: str | None) -> str:
        if conversation_id:
            return f"cid:{conversation_id}"
        if ctx.user_id:
            return f"user:{ctx.user_id}"
        if ctx.device_id:
            return f"device:{ctx.device_id}"
        return "default"

    async def _async_handle_timer_locally(
        self, *, ctx: ConversationContext, conversation_id: str | None, text: str
    ) -> str | None:
        timer_key = self._timer_key(ctx=ctx, conversation_id=conversation_id)
        kind = _timer_intent_kind(text)
        if kind is None:
            return None

        if kind == "cancel":
            if timer_key in self._timers:
                self._timers.pop(timer_key, None)
                return "Okay, canceled your timer."
            return "You don't have an active timer."

        if kind == "remaining":
            timer = self._timers.get(timer_key)
            if not timer:
                return "You don't have an active timer."

            ends_at = float(timer.get("ends_at", 0.0))
            remaining = int(round(ends_at - time.time()))
            if remaining <= 0:
                return "Your timer is done."

            mins, secs = divmod(remaining, 60)
            if mins > 0:
                return f"You have {mins} minute(s) and {secs} second(s) left on your timer."
            return f"You have {secs} second(s) left on your timer."

        if kind == "start":
            seconds = _parse_duration_seconds(text)
            if seconds is None:
                return "How long should I set the timer for?"

            now = time.time()
            ends_at = now + seconds
            self._timers[timer_key] = {
                "started_at": now,
                "ends_at": ends_at,
                "seconds": float(seconds),
            }
            self.hass.async_create_task(
                self._async_timer_finished(conversation_id=timer_key, ends_at=ends_at)
            )

            mins, secs = divmod(seconds, 60)
            if mins > 0 and secs:
                return f"Okay, starting a timer for {mins} minute(s) and {secs} second(s)."
            if mins > 0:
                return f"Okay, starting a timer for {mins} minute(s)."
            return f"Okay, starting a timer for {secs} second(s)."

        return None

    async def _async_local_timer_tool(
        self, *, ctx: ConversationContext, conversation_id: str | None, args: dict[str, Any]
    ) -> dict[str, Any]:
        timer_key = self._timer_key(ctx=ctx, conversation_id=conversation_id)
        action = str(args.get("action", "")).strip().lower()
        if action not in {"start", "cancel", "remaining"}:
            raise ValueError("local_timer.action must be start/cancel/remaining")

        if action == "cancel":
            if timer_key in self._timers:
                self._timers.pop(timer_key, None)
                return {"ok": True, "message": "Canceled."}
            return {"ok": False, "message": "No active timer."}

        if action == "remaining":
            timer = self._timers.get(timer_key)
            if not timer:
                return {"ok": False, "message": "No active timer."}
            ends_at = float(timer.get("ends_at", 0.0))
            remaining = int(round(ends_at - time.time()))
            return {"ok": True, "remaining_seconds": max(0, remaining)}

        seconds = int(args.get("seconds") or 0)
        if seconds <= 0:
            raise ValueError("local_timer.seconds must be > 0")
        now = time.time()
        ends_at = now + seconds
        self._timers[timer_key] = {
            "started_at": now,
            "ends_at": ends_at,
            "seconds": float(seconds),
        }
        self.hass.async_create_task(
            self._async_timer_finished(conversation_id=timer_key, ends_at=ends_at)
        )
        return {"ok": True, "ends_at": ends_at, "seconds": seconds}

    def _normalize_agent_step(self, obj: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize model output to {action: tool/final, ...}.

        Accepts common schema variants, including:
        - {"action":"tool","tool":"name","args":{...}}
        - {"action":"final","final_text":"..."}
        - {"action":"<tool_name>","args":{...}}  (shorthand)
        - {"tool":"name","args":{...}}          (missing action)
        """
        if not isinstance(obj, dict):
            return None

        action = str(obj.get("action") or "").strip()
        if action.lower() == "final":
            return {"action": "final", "final_text": str(obj.get("final_text") or "").strip()}

        if action.lower() == "tool":
            tool_name = str(obj.get("tool") or "").strip()
            args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
            if tool_name:
                return {"action": "tool", "tool": tool_name, "args": args}
            return None

        # Shorthand: action is the tool name.
        if action and action in self._tool_specs_by_name:
            args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
            return {"action": "tool", "tool": action, "args": args}

        # Missing action but has tool.
        tool_name = str(obj.get("tool") or "").strip()
        if tool_name and tool_name in self._tool_specs_by_name:
            args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
            return {"action": "tool", "tool": tool_name, "args": args}

        return None

    async def _async_memory_write_tool(self, *, ctx: ConversationContext, args: dict[str, Any]) -> dict[str, Any]:
        text = str(args.get("text", "")).strip()
        if not text:
            raise ValueError("memory_write.text is required")
        expires_days = args.get("expires_days")
        if expires_days is None or expires_days == "":
            expires_days_i = self._memory_expires_days
        else:
            expires_days_i = int(expires_days)

        emb = await self._embeddings.embed(text)
        item_id = await self._memory.remember(
            embedding=emb,
            text=text,
            ctx=ctx,
            expires_at=expiry_timestamp(days=expires_days_i),
        )
        return {"ok": True, "memory_id": item_id}

    async def _async_select_tools(self, *, query_embedding: list[float]) -> list[ToolSpec]:
        await self._async_index_tools()
        hits = await self._store.query(kind="tool", query_embedding=query_embedding, top_k=self._tool_top_k)
        selected: list[ToolSpec] = []
        for item, _score in hits:
            name = str(item.metadata.get("name") or "")
            spec = self._tool_specs_by_name.get(name)
            if spec is not None:
                selected.append(spec)
        # Ensure builtins exist even if similarity is low.
        for t in builtin_tools():
            if t.name not in {x.name for x in selected}:
                selected.append(t)
        return selected

    async def _async_agent_loop(
        self,
        *,
        ctx: ConversationContext,
        conversation_id: str,
        user_text: str,
    ) -> str:
        await self._async_ensure_store()
        await self._async_prune_if_needed()

        query_emb = await self._embeddings.embed(user_text)
        memories = await self._memory.recall(embedding=query_emb, ctx=ctx, top_k=self._memory_top_k)
        tools = await self._async_select_tools(query_embedding=query_emb)

        # Short-term buffer
        buffer_msgs = await self._buffer.read(ctx=ctx)
        if len(buffer_msgs) > self._buffer_max_turns * 2:
            # Best-effort cap. Keep newest turns.
            buffer_msgs = buffer_msgs[-(self._buffer_max_turns * 2) :]

        tools_block = tool_prompt_block(tools)
        memory_block = "\n".join(f"- {m}" for m in memories)

        system = (
            f"{AGENT_PROMPT}\n\n"
            f"Available tools:\n{tools_block}\n\n"
            f"Relevant memories:\n{memory_block if memory_block else '- (none)'}\n"
        )

        # Build working messages for the LLM.
        messages: list[dict[str, str]] = []
        for m in buffer_msgs:
            role = str(m.get("role") or "")
            if role in {"user", "assistant"}:
                messages.append({"role": role, "content": str(m.get("content") or "")})
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != user_text:
            messages.append({"role": "user", "content": user_text})

        tool_trace: list[dict[str, Any]] = []

        import aiohttp

        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for _step in range(max(1, self._agent_max_steps)):
                raw = await self._llm.generate(system=system, messages=messages)
                parsed = _extract_first_json_object(raw) or {}
                obj = self._normalize_agent_step(parsed)
                if obj is None:
                    # Ask the model to correct itself rather than showing tool JSON to the user.
                    messages.append(
                        {
                            "role": "user",
                            "content": "Your last response was invalid. Output STRICT JSON only matching one of: {action: tool, tool: name, args:{...}} or {action: final, final_text: ...}.",
                        }
                    )
                    continue

                action = str(obj.get("action") or "").strip().lower()
                if action == "final":
                    final_text = str(obj.get("final_text") or "").strip()
                    return final_text or "Sorry, I couldn't generate a response."

                tool_name = str(obj.get("tool") or "").strip()
                args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
                spec = self._tool_specs_by_name.get(tool_name)
                if spec is None:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                {"tool_error": f"Unknown tool '{tool_name}'"}, ensure_ascii=False
                            ),
                        }
                    )
                    continue

                try:
                    result = await execute_tool(
                        hass=self.hass,
                        tool=spec,
                        args=args,
                        allow_python=self._allow_python,
                        session=session,
                        local_timer_handler=lambda a: self._async_local_timer_tool(
                            ctx=ctx, conversation_id=conversation_id, args=a
                        ),
                        memory_write_handler=lambda a: self._async_memory_write_tool(ctx=ctx, args=a),
                    )
                except Exception as err:  # noqa: BLE001
                    result = {"ok": False, "error": str(err)}

                tool_trace.append({"tool": tool_name, "args": args, "result": result})
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"tool": tool_name, "result": result}, ensure_ascii=False
                        ),
                    }
                )

        # If we ran out of steps, provide a concise status.
        if tool_trace:
            return "I started working on that, but ran out of steps. Try asking again more specifically."
        return "Sorry, I couldn't complete that."

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: Any,
    ) -> ConversationResult:
        conversation_id = getattr(user_input, "conversation_id", None) or "default"

        ctx = ConversationContext(
            speaker=None,
            device_id=getattr(user_input, "device_id", None),
            user_id=getattr(getattr(user_input, "context", None), "user_id", None),
            conversation_id=getattr(user_input, "conversation_id", None),
            language=getattr(user_input, "language", None),
        )

        # Persist into buffer before generating the answer.
        try:
            await self._async_ensure_store()
            await self._buffer.append(ctx=ctx, role="user", content=user_input.text)
        except Exception:  # noqa: BLE001
            _LOGGER.debug("Failed writing buffer", exc_info=True)

        # If device_id is missing and the request looks like a timer, keep the old heuristic fallback.
        if user_input.device_id is None and _looks_like_timer_request(user_input.text):
            local_timer_response = await self._async_handle_timer_locally(
                ctx=ctx, conversation_id=getattr(user_input, "conversation_id", None), text=user_input.text
            )
            if local_timer_response is not None:
                response_text = local_timer_response
                response = IntentResponse(language=user_input.language)
                response.async_set_speech(response_text)
                return ConversationResult(
                    conversation_id=conversation_id,
                    response=response,
                    continue_conversation=False,
                )

        # Primary path: agent loop (tools + memory)
        try:
            # If provider is github_copilot and a base_url is provided,
            # delegate to the add-on's HTTP API instead of running locally.
            if (
                (self._llm_config.provider or "").lower() == "github_copilot"
                and (self._llm_config.base_url or "").strip()
            ):
                import aiohttp

                base = str(self._llm_config.base_url).rstrip("/")
                url = f"{base}/converse"
                headers = {"Content-Type": "application/json"}
                # Reuse llm_api_key as X-API-Key for the add-on if set.
                if self._llm_config.api_key:
                    headers["X-API-Key"] = self._llm_config.api_key

                payload = {
                    "text": user_input.text,
                    "conversation_id": conversation_id,
                    "language": getattr(user_input, "language", None),
                    "device_id": getattr(user_input, "device_id", None),
                    "user_id": getattr(getattr(user_input, "context", None), "user_id", None),
                }

                timeout = aiohttp.ClientTimeout(total=60)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            raise RuntimeError(f"Add-on API error ({resp.status}): {text[:200]}")
                        data = await resp.json()
                        response_text = str(data.get("text", "") or "Sorry, I couldn't generate a response.")
            else:
                response_text = await self._async_agent_loop(
                    ctx=ctx,
                    conversation_id=conversation_id,
                    user_text=user_input.text,
                )
        except Exception as err:  # noqa: BLE001
            _LOGGER.exception("Agent loop failed")
            response_text = f"Sorry, I couldn't complete that: {err}"

        response_text = response_text.strip() or "Sorry, I couldn't generate a response."

        # Persist assistant response into buffer
        try:
            await self._buffer.append(ctx=ctx, role="assistant", content=response_text)
        except Exception:  # noqa: BLE001
            _LOGGER.debug("Failed writing buffer", exc_info=True)

        # Add to chat log
        try:
            from homeassistant.components.conversation.chat_log import AssistantContent

            chat_log.async_add_assistant_content_without_tools(
                AssistantContent(agent_id=user_input.agent_id, content=response_text)
            )
        except Exception:  # noqa: BLE001
            _LOGGER.debug("Unable to add to chat log", exc_info=True)

        response = IntentResponse(language=user_input.language)
        response.async_set_speech(response_text)

        return ConversationResult(
            conversation_id=conversation_id,
            response=response,
            continue_conversation=False,
        )
