from __future__ import annotations

import logging
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

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: Any,
    ) -> ConversationResult:
        # 1) Prefer Home Assistant's built-in intent engine for device control.
        # This keeps timers/lights/etc working the same as the default Assist agent.
        try:
            from homeassistant.components.conversation.default_agent import DefaultAgent

            domain_data = self.hass.data.setdefault(DOMAIN, {})
            default_agent = domain_data.get("_ha_default_agent")
            if default_agent is None:
                default_agent = DefaultAgent(self.hass)
                domain_data["_ha_default_agent"] = default_agent

            intent_result = await default_agent._async_handle_message(user_input, chat_log)
            # If the default agent couldn't match any intent, fall back to the LLM.
            if (
                intent_result is not None
                and intent_result.response is not None
                and not (
                    getattr(intent_result.response, "response_type", None)
                    == ha_intent.IntentResponseType.ERROR
                    and getattr(intent_result.response, "error_code", None)
                    == ha_intent.IntentResponseErrorCode.NO_INTENT_MATCH
                )
            ):
                return intent_result
        except Exception:  # noqa: BLE001
            _LOGGER.debug("Default agent handling failed; falling back to LLM", exc_info=True)

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

        try:
            # Call LLM
            response_text = await self._llm.generate(
                system=SYSTEM_PROMPT,
                messages=history,
            )
            response_text = response_text.strip() or "Sorry, I couldn't generate a response."
        except Exception as err:
            _LOGGER.exception("LLM call failed")
            response_text = f"Sorry, I encountered an error: {err}"

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
