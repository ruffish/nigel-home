from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import aiohttp

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: str | None


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    async def generate(self, *, system: str, messages: list[dict[str, str]]) -> str:
        provider = self._config.provider.lower()

        if provider == "google":
            return await self._google_generate(system=system, messages=messages)
        if provider in {"openai", "openai_compatible"}:
            return await self._openai_generate(system=system, messages=messages)

        raise ValueError(f"Unknown LLM provider: {provider}")

    async def _google_generate(self, *, system: str, messages: list[dict[str, str]]) -> str:
        """Call Google Gemini API directly via HTTP."""
        # Build the prompt from messages
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(content)
        user_prompt = "\n".join(parts)

        # Google Gemini API endpoint
        # Accept either "gemini-2.5-flash" or "models/gemini-2.5-flash" (and strip any accidental ":..." suffix).
        model = (self._config.model or "").strip()
        if model.startswith("models/"):
            model = model[len("models/") :]
        if ":" in model:
            model = model.split(":", 1)[0]

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        generation_config = {
            "temperature": 0.7,
            "maxOutputTokens": 1024,
        }

        payload = {
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": generation_config,
        }

        headers = {
            "Content-Type": "application/json",
            # Prefer documented auth header; avoids query-string auth being blocked in some environments.
            "x-goog-api-key": self._config.api_key,
        }
        params = None

        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers, params=params) as resp:
                if resp.status != 200:
                    error_text = await resp.text()

                    # Some models (notably certain Gemma endpoints) do not allow developer/system instruction.
                    # If so, retry once by embedding the system prompt into the user content.
                    if (
                        resp.status == 400
                        and "Developer instruction is not enabled" in error_text
                        and system.strip()
                    ):
                        fallback_text = f"{system.strip()}\n\n{user_prompt}".strip()
                        fallback_payload = {
                            "contents": [{"role": "user", "parts": [{"text": fallback_text}]}],
                            "generationConfig": generation_config,
                        }
                        async with session.post(
                            url, json=fallback_payload, headers=headers, params=params
                        ) as resp2:
                            if resp2.status != 200:
                                error_text2 = await resp2.text()
                                _LOGGER.error(
                                    "Google API error (%s): %s", resp2.status, error_text2
                                )
                                raise RuntimeError(
                                    f"Google API error ({resp2.status}): {error_text2[:200]}"
                                )

                            data = await resp2.json()
                    else:
                        _LOGGER.error("Google API error (%s): %s", resp.status, error_text)
                        raise RuntimeError(
                            f"Google API error ({resp.status}): {error_text[:200]}"
                        )
                else:
                    data = await resp.json()

        # Extract text from response
        try:
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return str(parts[0].get("text", ""))
        except Exception as err:
            _LOGGER.error("Failed to parse Google response: %s", err)

        return ""

    async def _openai_generate(self, *, system: str, messages: list[dict[str, str]]) -> str:
        """Call OpenAI-compatible API directly via HTTP."""
        base_url = self._config.base_url or "https://api.openai.com/v1"
        url = f"{base_url.rstrip('/')}/chat/completions"

        req_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        req_messages.extend(messages)

        payload = {
            "model": self._config.model,
            "messages": req_messages,
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
        }

        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    _LOGGER.error("OpenAI API error (%s): %s", resp.status, error_text)
                    raise RuntimeError(f"OpenAI API error ({resp.status}): {error_text[:200]}")

                data = await resp.json()

        # Extract text from response
        try:
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return str(message.get("content", ""))
        except Exception as err:
            _LOGGER.error("Failed to parse OpenAI response: %s", err)

        return ""
