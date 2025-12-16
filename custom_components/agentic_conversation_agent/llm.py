from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

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
        try:
            import google.generativeai as genai
        except ImportError as err:
            raise RuntimeError("google-generativeai package not installed") from err

        # Flatten messages into a prompt
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
        prompt = "\n".join(parts)

        def _do() -> str:
            genai.configure(api_key=self._config.api_key)
            model = genai.GenerativeModel(self._config.model, system_instruction=system)
            resp = model.generate_content(prompt)
            return getattr(resp, "text", "") or ""

        return await asyncio.to_thread(_do)

    async def _openai_generate(self, *, system: str, messages: list[dict[str, str]]) -> str:
        try:
            from openai import OpenAI
        except ImportError as err:
            raise RuntimeError("openai package not installed") from err

        req_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        req_messages.extend(messages)

        def _do() -> str:
            client = OpenAI(
                api_key=self._config.api_key,
                base_url=self._config.base_url if self._config.base_url else None,
            )
            resp = client.chat.completions.create(
                model=self._config.model,
                messages=req_messages,  # type: ignore[arg-type]
                temperature=0.7,
            )
            return resp.choices[0].message.content or ""

        return await asyncio.to_thread(_do)
