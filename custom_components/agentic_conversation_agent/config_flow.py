from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult

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

_LOGGER = logging.getLogger(__name__)


class AgenticConversationAgentConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 2

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
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

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return AgenticConversationAgentOptionsFlowHandler(config_entry)


class AgenticConversationAgentOptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        # In newer Home Assistant versions, OptionsFlow may expose `config_entry`
        # as a read-only property. Avoid assigning to it.
        self._config_entry = config_entry

    @property
    def _entry(self) -> config_entries.ConfigEntry:
        """Return the current config entry."""
        # Prefer HA-provided property if present.
        return getattr(self, "config_entry", self._config_entry)

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        # Get current values from options, falling back to data (for migration/initial setup)
        current_provider = self._entry.options.get(
            CONF_LLM_PROVIDER,
            self._entry.data.get(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER)
        )
        current_model = self._entry.options.get(
            CONF_LLM_MODEL,
            self._entry.data.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL)
        )
        current_api_key = self._entry.options.get(
            CONF_LLM_API_KEY,
            self._entry.data.get(CONF_LLM_API_KEY, "")
        )
        current_base_url = self._entry.options.get(
            CONF_LLM_BASE_URL,
            self._entry.data.get(CONF_LLM_BASE_URL, "")
        )

        schema = vol.Schema(
            {
                vol.Required(CONF_LLM_PROVIDER, default=current_provider): vol.In(LLM_PROVIDERS),
                vol.Required(CONF_LLM_MODEL, default=current_model): str,
                vol.Required(CONF_LLM_API_KEY, default=current_api_key): str,
                vol.Optional(CONF_LLM_BASE_URL, default=current_base_url): str,
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema)
