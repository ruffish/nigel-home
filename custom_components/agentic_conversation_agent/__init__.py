from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import (
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_LLM_PROVIDER,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry to new version."""
    _LOGGER.debug("Migrating from version %s", config_entry.version)

    if config_entry.version == 1:
        # Old format had base_url, api_key, timeout - we need to ask user to reconfigure
        # Since we can't know their LLM settings, just create a placeholder
        new_data = {
            "name": config_entry.data.get("name", "Agentic Conversation Agent"),
            CONF_LLM_PROVIDER: "google",
            CONF_LLM_MODEL: "gemini-1.5-flash",
            CONF_LLM_API_KEY: "",  # User will need to reconfigure
            CONF_LLM_BASE_URL: "",
        }

        hass.config_entries.async_update_entry(
            config_entry, data=new_data, version=2
        )
        _LOGGER.info(
            "Migration to version 2 successful. Please reconfigure your API key in the integration settings."
        )

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Agentic Conversation Agent from a config entry."""
    config = {**entry.data, **entry.options}
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = config

    await hass.config_entries.async_forward_entry_setups(entry, ["conversation"])

    entry.async_on_unload(entry.add_update_listener(update_listener))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    unload_ok = await hass.config_entries.async_unload_platforms(entry, ["conversation"])
    if unload_ok:
        hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
    return unload_ok


async def update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Update listener."""
    await hass.config_entries.async_reload(entry.entry_id)
