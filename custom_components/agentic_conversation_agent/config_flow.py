from __future__ import annotations

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME

from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_TIMEOUT,
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    DOMAIN,
)


class AgenticConversationAgentConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input=None):
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        errors = {}

        if user_input is not None:
            return self.async_create_entry(
                title=user_input.get(CONF_NAME, "Agentic Conversation Agent"),
                data={
                    CONF_BASE_URL: user_input[CONF_BASE_URL],
                    CONF_API_KEY: user_input.get(CONF_API_KEY, ""),
                    CONF_TIMEOUT: user_input.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                    CONF_NAME: user_input.get(CONF_NAME, "Agentic Conversation Agent"),
                },
            )

        schema = vol.Schema(
            {
                vol.Optional(CONF_NAME, default="Agentic Conversation Agent"): str,
                vol.Required(CONF_BASE_URL, default=DEFAULT_BASE_URL): str,
                vol.Optional(CONF_API_KEY, default=""): str,
                vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): vol.Coerce(int),
            }
        )

        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)
