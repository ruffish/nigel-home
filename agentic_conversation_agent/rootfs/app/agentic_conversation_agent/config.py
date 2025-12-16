from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str | None
    base_url: str | None


@dataclass(frozen=True)
class EmbeddingsConfig:
    provider: str
    model: str | None
    api_key: str | None
    base_url: str | None


@dataclass(frozen=True)
class RAGConfig:
    tools_top_n: int


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int


@dataclass(frozen=True)
class MemoryConfig:
    top_k: int
    buffer_max_turns: int
    expiry_days_default: int
    summarization_model: str | None


@dataclass(frozen=True)
class ToolsConfig:
    dir: str
    allow_python: bool


@dataclass(frozen=True)
class AppConfig:
    log_level: str
    listen_host: str
    listen_port: int
    api_key: str | None
    llm: LLMConfig
    embeddings: EmbeddingsConfig
    rag: RAGConfig
    agent: AgentConfig
    memory: MemoryConfig
    tools: ToolsConfig
    db_path: str


def load_options(options_path: str = "/data/options.json") -> dict:
    path = Path(options_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_config() -> AppConfig:
    opts = load_options()

    llm_opts = opts.get("llm", {})
    embeddings_opts = opts.get("embeddings", {})
    rag_opts = opts.get("rag", {})
    agent_opts = opts.get("agent", {})
    memory_opts = opts.get("memory", {})
    tools_opts = opts.get("tools", {})

    return AppConfig(
        log_level=str(opts.get("log_level", "info")),
        listen_host=str(opts.get("listen_host", "0.0.0.0")),
        listen_port=int(opts.get("listen_port", 8099)),
        api_key=(opts.get("api_key") or None),
        llm=LLMConfig(
            provider=str(llm_opts.get("provider", "google")),
            model=str(llm_opts.get("model", "gemini-1.5-flash")),
            api_key=(llm_opts.get("api_key") or None),
            base_url=(llm_opts.get("base_url") or None),
        ),
        embeddings=EmbeddingsConfig(
            provider=str(embeddings_opts.get("provider", "simple")),
            model=(embeddings_opts.get("model") or None),
            api_key=(embeddings_opts.get("api_key") or None),
            base_url=(embeddings_opts.get("base_url") or None),
        ),
        rag=RAGConfig(
            tools_top_n=int(rag_opts.get("tools_top_n", 8)),
        ),
        agent=AgentConfig(
            max_steps=int(agent_opts.get("max_steps", 6)),
        ),
        memory=MemoryConfig(
            top_k=int(memory_opts.get("top_k", 5)),
            buffer_max_turns=int(memory_opts.get("buffer_max_turns", 12)),
            expiry_days_default=int(memory_opts.get("expiry_days_default", 30)),
            summarization_model=(memory_opts.get("summarization_model") or None),
        ),
        tools=ToolsConfig(
            dir=str(tools_opts.get("dir", "/data/tools")),
            allow_python=bool(tools_opts.get("allow_python", False)),
        ),
        db_path=str(opts.get("db_path", "/data/agent.db")),
    )
