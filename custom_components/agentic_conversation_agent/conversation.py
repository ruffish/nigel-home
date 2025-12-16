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
    DOMAIN,
)
from .llm import LLMClient, LLMConfig

_LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful Home Assistant voice assistant named Nigel.
You help users control their smart home and answer questions.
Be concise and friendly. If you don't know something, say so.
When the user asks to control devices, explain what you would do (you don't have direct device control yet)."""


ROUTER_PROMPT = """You are an AI assistant for Home Assistant.

You can either:
1) Ask Home Assistant to handle the request (device control, timers, queries about entity states), OR
2) Respond normally (general questions / chit-chat).

Output MUST be STRICT JSON only (no markdown, no extra text), matching ONE of:

{ "action": "hass" }
  - Use this when Home Assistant should execute/answer via its built-in assistant capabilities.

{ "action": "final", "final_text": "..." }
  - Use this when you should answer directly.

Guidance:
- For commands like "start a 5 minute timer", "turn on the kitchen lights", "how long left on my timer", choose action=hass.
- Only choose action=final for questions that aren't about the home.
"""


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

        # Simple conversation history (in-memory)
        self._history: dict[str, list[dict[str, str]]] = {}

        # Simple in-integration timers (for clients that don't provide a device_id).
        # Keyed by conversation_id.
        self._timers: dict[str, dict[str, float]] = {}

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

    async def _async_handle_timer_locally(
        self, *, conversation_id: str, text: str
    ) -> str | None:
        kind = _timer_intent_kind(text)
        if kind is None:
            return None

        if kind == "cancel":
            if conversation_id in self._timers:
                self._timers.pop(conversation_id, None)
                return "Okay, canceled your timer."
            return "You don't have an active timer."

        if kind == "remaining":
            timer = self._timers.get(conversation_id)
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
            self._timers[conversation_id] = {
                "started_at": now,
                "ends_at": ends_at,
                "seconds": float(seconds),
            }
            self.hass.async_create_task(
                self._async_timer_finished(conversation_id=conversation_id, ends_at=ends_at)
            )

            mins, secs = divmod(seconds, 60)
            if mins > 0 and secs:
                return f"Okay, starting a timer for {mins} minute(s) and {secs} second(s)."
            if mins > 0:
                return f"Okay, starting a timer for {mins} minute(s)."
            return f"Okay, starting a timer for {secs} second(s)."

        return None

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: Any,
    ) -> ConversationResult:
        conversation_id = getattr(user_input, "conversation_id", None) or "default"

        # Get or create conversation history
        if conversation_id not in self._history:
            self._history[conversation_id] = []

        history = self._history[conversation_id]

        # Add user message to history
        history.append({"role": "user", "content": user_input.text})

        # Keep only last 10 messages
        if len(history) > 10:
            history = history[-10:]
            self._history[conversation_id] = history

        # Let the LLM decide whether to call Home Assistant (tools/intents) or answer directly.
        try:
            router_raw = await self._llm.generate(system=ROUTER_PROMPT, messages=history)
            router = _extract_first_json_object(router_raw) or {}
        except Exception:  # noqa: BLE001
            _LOGGER.debug("Router call failed; defaulting to hass", exc_info=True)
            router = {"action": "hass"}

        action = str(router.get("action", "")).strip().lower()

        if action == "hass":
            # Local fallback for timers when the client doesn't provide a device_id.
            # The built-in timer intent expects a device that supports timers.
            if user_input.device_id is None and _looks_like_timer_request(user_input.text):
                local_timer_response = await self._async_handle_timer_locally(
                    conversation_id=conversation_id, text=user_input.text
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

            try:
                from homeassistant.components.conversation.default_agent import DefaultAgent

                domain_data = self.hass.data.setdefault(DOMAIN, {})
                default_agent = domain_data.get("_ha_default_agent")
                if default_agent is None:
                    default_agent = DefaultAgent(self.hass)
                    domain_data["_ha_default_agent"] = default_agent

                result = await default_agent._async_handle_message(user_input, chat_log)

                # Mirror the speech into our in-memory history so follow-ups have context.
                try:
                    speech = result.response.speech.get("plain", {}).get("speech", "")
                except Exception:
                    speech = ""
                if speech:
                    history.append({"role": "assistant", "content": speech})

                return result
            except Exception as err:  # noqa: BLE001
                _LOGGER.exception("Home Assistant action handling failed")
                response_text = f"Sorry, I couldn't execute that in Home Assistant: {err}"
        else:
            response_text = str(router.get("final_text") or "").strip()
            if not response_text:
                # Fall back to a normal LLM reply if the router returned an invalid schema.
                try:
                    response_text = await self._llm.generate(
                        system=SYSTEM_PROMPT,
                        messages=history,
                    )
                    response_text = response_text.strip()
                except Exception as err:  # noqa: BLE001
                    _LOGGER.exception("LLM call failed")
                    response_text = f"Sorry, I encountered an error: {err}"

        response_text = response_text.strip() or "Sorry, I couldn't generate a response."

        # Add assistant response to history
        history.append({"role": "assistant", "content": response_text})

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
