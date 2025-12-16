from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # noqa: BLE001
    genai = None

from .utils import safe_json_loads


_LOGGER = logging.getLogger(__name__)


class LLMClient(Protocol):
    async def next_action(self, *, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class OpenAIChatLLM:
    api_key: str
    model: str
    base_url: str | None = None

    async def next_action(self, *, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        req_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        req_messages.extend(messages)

        def _do(with_json_mode: bool) -> str:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": req_messages,
                "temperature": 0.2,
            }
            if with_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
            return resp.choices[0].message.content or "{}"

        try:
            content = await asyncio.to_thread(_do, True)
        except Exception:  # noqa: BLE001
            _LOGGER.debug("OpenAI JSON mode failed; retrying without response_format", exc_info=True)
            content = await asyncio.to_thread(_do, False)

        return safe_json_loads(content)


@dataclass(frozen=True)
class GoogleChatLLM:
    api_key: str
    model: str

    async def next_action(self, *, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        if genai is None:
            raise RuntimeError("google-generativeai is not installed")

        # Flatten into a single text prompt to keep behavior consistent.
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(content)
        prompt = "\n".join(parts)

        def _do() -> str:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model, system_instruction=system)
            resp = model.generate_content(prompt)
            return getattr(resp, "text", None) or ""

        text = await asyncio.to_thread(_do)

        # Be tolerant of code fences.
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        return safe_json_loads(text)


def build_llm_client(*, provider: str, model: str, api_key: str | None, base_url: str | None) -> LLMClient:
    provider = (provider or "openai").lower()
    if provider in {"openai", "openai_compatible"}:
        if not api_key:
            raise ValueError("LLM api_key is required for OpenAI/openai_compatible")
        return OpenAIChatLLM(api_key=api_key, model=model, base_url=base_url)
    if provider == "google":
        if not api_key:
            raise ValueError("LLM api_key is required for Google")
        return GoogleChatLLM(api_key=api_key, model=model)
    raise ValueError(f"Unknown LLM provider: {provider}")
