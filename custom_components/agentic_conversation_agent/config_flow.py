from __future__ import annotations

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME

from .const import (
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DOMAIN,
    LLM_PROVIDERS,
)


class AgenticConversationAgentConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 2

    async def async_step_user(self, user_input=None):
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        errors = {}

        if user_input is not None:
            # Validate API key is provided
            if not user_input.get(CONF_LLM_API_KEY):
                errors[CONF_LLM_API_KEY] = "api_key_required"
            else:
                return self.async_create_entry(
                    title=user_input.get(CONF_NAME, "Agentic Conversation Agent"),
                    data={
                        CONF_NAME: user_input.get(CONF_NAME, "Agentic Conversation Agent"),
                        CONF_LLM_PROVIDER: user_input.get(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER),
                        CONF_LLM_MODEL: user_input.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL),
                        CONF_LLM_API_KEY: user_input.get(CONF_LLM_API_KEY, ""),
                        CONF_LLM_BASE_URL: user_input.get(CONF_LLM_BASE_URL, ""),
                    },
                )

        schema = vol.Schema(
            {
                vol.Optional(CONF_NAME, default="Agentic Conversation Agent"): str,
                vol.Required(CONF_LLM_PROVIDER, default=DEFAULT_LLM_PROVIDER): vol.In(LLM_PROVIDERS),
                vol.Required(CONF_LLM_MODEL, default=DEFAULT_LLM_MODEL): str,
                vol.Required(CONF_LLM_API_KEY): str,
                vol.Optional(CONF_LLM_BASE_URL, default=""): str,
            }
        )

        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)
