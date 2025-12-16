from __future__ import annotations

import asyncio
import logging
import subprocess
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


@dataclass(frozen=True)
class GitHubCopilotCLI:
    """Use GitHub Copilot CLI as the LLM backend."""
    
    model: str = "gpt-4o"  # Model is informational; gh copilot uses your subscription

    async def next_action(self, *, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Call GitHub Copilot CLI to get a response."""
        # Build a comprehensive prompt from system + messages
        parts: list[str] = []
    if provider == "github_copilot":
        # No API key needed - uses gh CLI authentication
        return GitHubCopilotCLI(model=model or "gpt-4o")
        if system:
            parts.append(f"System: {system}")
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(content)
        
        full_prompt = "\n".join(parts)
        
        # Add explicit JSON formatting instruction
        full_prompt += "\n\nRespond in valid JSON format only."

        def _do() -> str:
            try:
                # Use GitHub Copilot CLI with -i flag for inline prompt
                result = subprocess.run(
                    ["copilot", "-i", full_prompt],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
                
                # If that fails, try piping to stdin
                result = subprocess.run(
                    ["copilot"],
                    input=full_prompt,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
                
                # Log error if available
                if result.stderr:
                    _LOGGER.error(f"GitHub Copilot CLI error: {result.stderr}")
                
                raise RuntimeError(f"GitHub Copilot CLI failed with exit code {result.returncode}")
                
            except FileNotFoundError:
                _LOGGER.error("GitHub Copilot CLI not found. Make sure 'copilot' command is installed and in PATH")
                raise RuntimeError("GitHub Copilot CLI not installed. Install from: https://github.com/cli/cli")
            except subprocess.TimeoutExpired:
                _LOGGER.error("GitHub Copilot CLI timed out")
                raise RuntimeError("GitHub Copilot CLI timeout")
            except Exception as e:
                _LOGGER.error(f"GitHub Copilot CLI error: {e}")
                raise

        text = await asyncio.to_thread(_do)
        
        # Clean up the response - gh copilot may add formatting
        text = text.strip()
        
        # Remove any markdown code fences
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        
        # Try to find JSON in the response if it's embedded in text
        if not text.startswith("{"):
            # Look for JSON object in the text
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx != -1:
                text = text[start_idx:end_idx + 1]
        
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
