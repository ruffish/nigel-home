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
    CONF_INDEX_HA_SERVICES,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_EMBEDDINGS_PROVIDER,
    DEFAULT_EMBEDDINGS_MODEL,
    DEFAULT_TOOLS_DIR,
    DEFAULT_ALLOW_PYTHON,
    DEFAULT_AGENT_MAX_STEPS,
    DEFAULT_TOOL_TOP_K,
    DEFAULT_MEMORY_TOP_K,
    DEFAULT_MEMORY_EXPIRES_DAYS,
    DEFAULT_BUFFER_MAX_TURNS,
    DEFAULT_INDEX_HA_SERVICES,
    DOMAIN,
    EMBEDDINGS_PROVIDERS,
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
            # Validate API key only for providers that require it
            provider = user_input.get(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER)
            api_key = user_input.get(CONF_LLM_API_KEY, "")
            if provider != "github_copilot" and not api_key:
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
                # Make API key optional at schema level; we validate conditionally above
                vol.Optional(CONF_LLM_API_KEY, default=""): str,
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

        current_embeddings_provider = self._entry.options.get(
            CONF_EMBEDDINGS_PROVIDER,
            self._entry.data.get(CONF_EMBEDDINGS_PROVIDER, DEFAULT_EMBEDDINGS_PROVIDER),
        )
        current_embeddings_model = self._entry.options.get(
            CONF_EMBEDDINGS_MODEL,
            self._entry.data.get(CONF_EMBEDDINGS_MODEL, DEFAULT_EMBEDDINGS_MODEL),
        )
        current_embeddings_api_key = self._entry.options.get(
            CONF_EMBEDDINGS_API_KEY,
            self._entry.data.get(CONF_EMBEDDINGS_API_KEY, ""),
        )
        current_embeddings_base_url = self._entry.options.get(
            CONF_EMBEDDINGS_BASE_URL,
            self._entry.data.get(CONF_EMBEDDINGS_BASE_URL, ""),
        )

        current_tools_dir = self._entry.options.get(
            CONF_TOOLS_DIR,
            self._entry.data.get(CONF_TOOLS_DIR, DEFAULT_TOOLS_DIR),
        )
        current_allow_python = self._entry.options.get(
            CONF_ALLOW_PYTHON,
            self._entry.data.get(CONF_ALLOW_PYTHON, DEFAULT_ALLOW_PYTHON),
        )
        current_agent_max_steps = self._entry.options.get(
            CONF_AGENT_MAX_STEPS,
            self._entry.data.get(CONF_AGENT_MAX_STEPS, DEFAULT_AGENT_MAX_STEPS),
        )
        current_tool_top_k = self._entry.options.get(
            CONF_TOOL_TOP_K,
            self._entry.data.get(CONF_TOOL_TOP_K, DEFAULT_TOOL_TOP_K),
        )
        current_memory_top_k = self._entry.options.get(
            CONF_MEMORY_TOP_K,
            self._entry.data.get(CONF_MEMORY_TOP_K, DEFAULT_MEMORY_TOP_K),
        )
        current_memory_expires_days = self._entry.options.get(
            CONF_MEMORY_EXPIRES_DAYS,
            self._entry.data.get(CONF_MEMORY_EXPIRES_DAYS, DEFAULT_MEMORY_EXPIRES_DAYS),
        )
        current_buffer_max_turns = self._entry.options.get(
            CONF_BUFFER_MAX_TURNS,
            self._entry.data.get(CONF_BUFFER_MAX_TURNS, DEFAULT_BUFFER_MAX_TURNS),
        )

        current_index_ha_services = self._entry.options.get(
            CONF_INDEX_HA_SERVICES,
            self._entry.data.get(CONF_INDEX_HA_SERVICES, DEFAULT_INDEX_HA_SERVICES),
        )

        schema = vol.Schema(
            {
                vol.Required(CONF_LLM_PROVIDER, default=current_provider): vol.In(LLM_PROVIDERS),
                vol.Required(CONF_LLM_MODEL, default=current_model): str,
            # Optional to support github_copilot which doesn't need a key
            vol.Optional(CONF_LLM_API_KEY, default=current_api_key): str,
                vol.Optional(CONF_LLM_BASE_URL, default=current_base_url): str,

                vol.Required(CONF_EMBEDDINGS_PROVIDER, default=current_embeddings_provider): vol.In(EMBEDDINGS_PROVIDERS),
                vol.Optional(CONF_EMBEDDINGS_MODEL, default=current_embeddings_model): str,
                vol.Optional(CONF_EMBEDDINGS_API_KEY, default=current_embeddings_api_key): str,
                vol.Optional(CONF_EMBEDDINGS_BASE_URL, default=current_embeddings_base_url): str,

                vol.Optional(CONF_TOOLS_DIR, default=current_tools_dir): str,
                vol.Required(CONF_ALLOW_PYTHON, default=current_allow_python): bool,
                vol.Required(CONF_AGENT_MAX_STEPS, default=current_agent_max_steps): vol.Coerce(int),
                vol.Required(CONF_TOOL_TOP_K, default=current_tool_top_k): vol.Coerce(int),
                vol.Required(CONF_MEMORY_TOP_K, default=current_memory_top_k): vol.Coerce(int),
                vol.Required(CONF_MEMORY_EXPIRES_DAYS, default=current_memory_expires_days): vol.Coerce(int),
                vol.Required(CONF_BUFFER_MAX_TURNS, default=current_buffer_max_turns): vol.Coerce(int),
                vol.Required(CONF_INDEX_HA_SERVICES, default=current_index_ha_services): bool,
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema)
