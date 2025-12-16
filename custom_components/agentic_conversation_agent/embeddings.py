from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Protocol

import aiohttp

_LOGGER = logging.getLogger(__name__)


class EmbeddingsProvider(Protocol):
    async def embed(self, text: str) -> list[float]:
        ...


def _simple_embed(text: str, dim: int = 256) -> list[float]:
    # Deterministic local fallback: hashed bag-of-bytes into a fixed vector.
    b = text.encode("utf-8", errors="ignore")
    vec = [0.0] * dim
    digest = hashlib.sha256(b).digest()
    seed = int.from_bytes(digest[:8], "big")
    for i, by in enumerate(b[:4096]):
        idx = (seed + i * 131 + by) % dim
        vec[idx] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


@dataclass(frozen=True)
class SimpleEmbeddings:
    async def embed(self, text: str) -> list[float]:
        return _simple_embed(text)


@dataclass(frozen=True)
class OllamaEmbeddings:
    base_url: str
    model: str

    async def embed(self, text: str) -> list[float]:
        url = self.base_url.rstrip("/") + "/api/embeddings"
        payload = {"model": self.model, "prompt": text}

        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                data = await resp.json(content_type=None)
                if resp.status >= 300:
                    raise RuntimeError(f"Ollama embeddings failed ({resp.status}): {data}")

        emb = data.get("embedding")
        if not isinstance(emb, list):
            raise RuntimeError("Ollama embeddings response missing 'embedding'")
        return [float(x) for x in emb]


@dataclass(frozen=True)
class OpenAICompatibleEmbeddings:
    api_key: str
    model: str
    base_url: str

    async def embed(self, text: str) -> list[float]:
        url = self.base_url.rstrip("/") + "/embeddings"
        payload = {"model": self.model, "input": [text]}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                data = await resp.json(content_type=None)
                if resp.status >= 300:
                    raise RuntimeError(f"Embeddings failed ({resp.status}): {data}")

        try:
            emb = data["data"][0]["embedding"]
        except Exception as err:  # noqa: BLE001
            raise RuntimeError(f"Unexpected embeddings response: {data}") from err

        if not isinstance(emb, list):
            raise RuntimeError("Embeddings response missing embedding list")
        return [float(x) for x in emb]


@dataclass(frozen=True)
class GoogleEmbeddings:
    api_key: str
    model: str

    async def embed(self, text: str) -> list[float]:
        # Gemini embeddings REST. Model usually: "text-embedding-004" or "models/text-embedding-004".
        model = (self.model or "").strip() or "text-embedding-004"
        if model.startswith("models/"):
            model = model[len("models/") :]
        if ":" in model:
            model = model.split(":", 1)[0]

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        payload = {"content": {"parts": [{"text": text}]}}

        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                data = await resp.json(content_type=None)
                if resp.status >= 300:
                    raise RuntimeError(f"Google embeddings failed ({resp.status}): {data}")

        emb = data.get("embedding") or data.get("embeddings")
        if isinstance(emb, dict):
            values = emb.get("values")
        else:
            values = None

        if not isinstance(values, list):
            raise RuntimeError(f"Unexpected Google embeddings response: {data}")
        return [float(x) for x in values]


def build_embeddings_provider(
    *,
    provider: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
) -> EmbeddingsProvider:
    provider = (provider or "simple").lower()

    if provider == "simple":
        return SimpleEmbeddings()

    if provider == "ollama":
        return OllamaEmbeddings(base_url=base_url or "http://localhost:11434", model=model or "nomic-embed-text")

    if provider in {"openai", "openai_compatible"}:
        if not api_key:
            raise ValueError("Embeddings api_key is required for openai/openai_compatible")
        # For OpenAI this should be "https://api.openai.com/v1". For local, point to your server.
        return OpenAICompatibleEmbeddings(
            api_key=api_key,
            model=model or "text-embedding-3-small",
            base_url=(base_url or "https://api.openai.com/v1"),
        )

    if provider == "google":
        if not api_key:
            raise ValueError("Embeddings api_key is required for google")
        return GoogleEmbeddings(api_key=api_key, model=model or "text-embedding-004")

    raise ValueError(f"Unknown embeddings provider: {provider}")
