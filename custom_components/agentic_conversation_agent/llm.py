from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Any

import aiohttp
import asyncio

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
        if provider == "github_copilot":
            return await self._github_copilot_generate(system=system, messages=messages)

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

    async def _github_copilot_generate(self, *, system: str, messages: list[dict[str, str]]) -> str:
        """Call GitHub Copilot via HTTP proxy when base_url is set, otherwise shell out to the CLI."""
        # Build a comprehensive prompt from system + messages
        parts: list[str] = []
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

        # Prefer an HTTP proxy if provided (e.g., the standalone copilot_server.py).
        if (self._config.base_url or "").strip():
            base = str(self._config.base_url).rstrip("/")
            # If the caller already included /copilot leave it; otherwise append.
            if not base.endswith("/copilot"):
                url = f"{base}/copilot"
            else:
                url = base

            headers = {"Content-Type": "application/json"}
            if self._config.api_key:
                # Optional key passthrough for proxies that enforce auth
                headers["X-API-Key"] = self._config.api_key

            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json={"prompt": full_prompt}, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"Copilot proxy error ({resp.status}): {error_text[:200]}")
                    data = await resp.json()
                    if not data.get("ok"):
                        raise RuntimeError(str(data.get("error") or "Copilot proxy returned an error"))
                    text = str(data.get("response", "")).strip()
                    if text:
                        return text

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

        # Clean up the response
        text = text.strip()

        # Remove any markdown code fences if present
        if text.startswith("```"):
            text = text.strip("`")
            # Skip language identifier if present
            lines = text.split("\n", 1)
            if len(lines) > 1:
                text = lines[1]

        return text

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
