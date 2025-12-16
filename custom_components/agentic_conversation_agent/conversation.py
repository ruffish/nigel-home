from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import aiohttp

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from homeassistant.components import intent
from homeassistant.components.conversation import (
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
)

if TYPE_CHECKING:
    from homeassistant.components.conversation.chat_log import ChatLog

from .const import CONF_API_KEY, CONF_BASE_URL, CONF_TIMEOUT, DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    async_add_entities([AgenticConversationEntity(hass, entry)])


class AgenticConversationEntity(ConversationEntity):
    _attr_supported_languages = "*"
    _attr_supported_features = ConversationEntityFeature.CONTROL

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._entry = entry
        self._attr_name = entry.data.get("name") or "Agentic Conversation Agent"
        self._attr_unique_id = f"{entry.entry_id}"
        self._base_url = str(entry.data.get(CONF_BASE_URL) or "")
        self._api_key = str(entry.data.get(CONF_API_KEY) or "")
        self._timeout = int(entry.data.get(CONF_TIMEOUT) or 30)

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: Any,
    ) -> ConversationResult:
        device_id = getattr(user_input, "device_id", None)
        language = getattr(user_input, "language", None)
        conversation_id = getattr(user_input, "conversation_id", None)

        speaker = None
        try:
            speaker = getattr(user_input.context, "user_id", None)
        except Exception:  # noqa: BLE001
            speaker = None

        payload = {
            "text": user_input.text,
            "language": language,
            "conversation_id": conversation_id,
            "device_id": device_id,
            "speaker": speaker,
            "user_id": speaker,
        }

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key

        url = self._base_url.rstrip("/") + "/converse"

        timeout = aiohttp.ClientTimeout(total=self._timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                data = await resp.json(content_type=None)

        text = str(data.get("text", "")).strip() or "Sorry, I couldn't generate a response."

        # Add to chat log
        try:
            from homeassistant.components.conversation.chat_log import AssistantContent

            chat_log.async_add_assistant_content_without_tools(
                AssistantContent(agent_id=user_input.agent_id, content=text)
            )
        except Exception:  # noqa: BLE001
            _LOGGER.debug("Unable to add to chat log", exc_info=True)

        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(text)

        return ConversationResult(
            conversation_id=data.get("conversation_id"),
            response=response,
            continue_conversation=bool(data.get("continue_conversation", False)),
        )
