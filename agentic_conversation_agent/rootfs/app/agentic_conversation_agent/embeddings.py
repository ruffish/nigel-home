from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import Protocol

import aiohttp
from openai import OpenAI

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # noqa: BLE001
    genai = None


_LOGGER = logging.getLogger(__name__)


class EmbeddingsProvider(Protocol):
    async def embed(self, text: str) -> list[float]:
        ...


def _simple_embed(text: str, dim: int = 256) -> list[float]:
    # Deterministic local fallback: hashed bag-of-bytes into a fixed vector.
    # This is not "smart" embeddings, but keeps RAG functional offline.
    b = text.encode("utf-8", errors="ignore")
    vec = [0.0] * dim
    digest = hashlib.sha256(b).digest()
    # seed
    seed = int.from_bytes(digest[:8], "big")
    for i, by in enumerate(b[:4096]):
        idx = (seed + i * 131 + by) % dim
        vec[idx] += 1.0
    # L2 normalize
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


@dataclass(frozen=True)
class SimpleEmbeddings:
    async def embed(self, text: str) -> list[float]:
        return _simple_embed(text)


@dataclass(frozen=True)
class OpenAIEmbeddings:
    api_key: str
    model: str
    base_url: str | None = None

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("OpenAI embeddings api_key is required")
        if not self.model:
            raise ValueError("OpenAI embeddings model is required")

    async def embed(self, text: str) -> list[float]:
        def _do() -> list[float]:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            resp = client.embeddings.create(model=self.model, input=[text])
            return list(resp.data[0].embedding)

        return await asyncio.to_thread(_do)


@dataclass(frozen=True)
class GoogleEmbeddings:
    api_key: str
    model: str

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("Google embeddings api_key is required")
        if not self.model:
            raise ValueError("Google embeddings model is required")
        if genai is None:
            raise ValueError("google-generativeai is not available")

    async def embed(self, text: str) -> list[float]:
        def _do() -> list[float]:
            genai.configure(api_key=self.api_key)
            # google-generativeai expects model like "models/text-embedding-004"
            resp = genai.embed_content(model=self.model, content=text)
            emb = resp["embedding"] if isinstance(resp, dict) else getattr(resp, "embedding", None)
            if emb is None:
                _LOGGER.debug("Unexpected embeddings response: %s", resp)
                raise RuntimeError("Google embeddings returned no embedding")
            return list(emb)

        return await asyncio.to_thread(_do)


def build_embeddings_provider(*, provider: str, model: str | None, api_key: str | None, base_url: str | None) -> EmbeddingsProvider:
    provider = (provider or "simple").lower()
    if provider == "simple":
        return SimpleEmbeddings()
    if provider == "ollama":
        return OllamaEmbeddings(base_url=base_url or "http://ollama:11434", model=model or "nomic-embed-text")
    if provider in {"openai", "openai_compatible"}:
        return OpenAIEmbeddings(api_key=api_key or "", model=model or "", base_url=base_url)
    if provider == "google":
        return GoogleEmbeddings(api_key=api_key or "", model=model or "models/text-embedding-004")
    raise ValueError(f"Unknown embeddings provider: {provider}")


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
