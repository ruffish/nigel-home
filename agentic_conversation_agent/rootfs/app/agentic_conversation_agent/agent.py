from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import aiohttp

from .config import AppConfig
from .embeddings import EmbeddingsProvider
from .llm import LLMClient
from .memory import BufferStore, ConversationContext, MemoryManager
from .tools import ToolSpec, execute_tool, tool_prompt_block
from .utils import UserVisibleError
from .vector_store import VectorStoreSQLite


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentResult:
    text: str
    continue_conversation: bool
    conversation_id: str | None


def _system_prompt(*, tool_block: str, memories: list[str]) -> str:
    mem_block = "\n".join([f"- {m}" for m in memories]) if memories else "(none)"

    return (
        "You are a Home Assistant Conversation Agent. "
        "You can answer normally, or choose a tool to control Home Assistant or fetch information.\n\n"
        "You MUST respond with STRICT JSON only (no markdown, no code fences).\n\n"
        "Output schema:\n"
        "{\n"
        "  \"action\": \"tool\" | \"final\",\n"
        "  \"tool_name\": string,               # required if action=tool\n"
        "  \"tool_args\": object,               # required if action=tool\n"
        "  \"final_text\": string,              # required if action=final\n"
        "  \"continue_conversation\": boolean   # optional\n"
        "}\n\n"
        "If you use tools: choose the SINGLE best next tool; you may call multiple tools across steps.\n"
        "Do not invent tool names. If no tool is needed, use action=final.\n\n"
        "Relevant long-term memories (may be empty):\n"
        f"{mem_block}\n\n"
        "Available tools:\n"
        f"{tool_block}\n"
    )


class Agent:
    def __init__(
        self,
        *,
        config: AppConfig,
        llm: LLMClient,
        summarizer_llm: LLMClient | None,
        embeddings: EmbeddingsProvider,
        store: VectorStoreSQLite,
        tools: list[ToolSpec],
        memory: MemoryManager,
        buffer: BufferStore,
    ) -> None:
        self._config = config
        self._llm = llm
        self._summarizer_llm = summarizer_llm or llm
        self._embeddings = embeddings
        self._store = store
        self._tools = {t.name: t for t in tools}
        self._memory = memory
        self._buffer = buffer

    async def reload_tools(self, *, tools: list[ToolSpec]) -> None:
        self._tools = {t.name: t for t in tools}

    async def converse(
        self,
        *,
        text: str,
        ctx: ConversationContext,
    ) -> AgentResult:
        pruned = self._memory.prune()
        if pruned:
            _LOGGER.debug("Pruned %s expired memories", pruned)

        # Embed user query
        query_emb = await self._embeddings.embed(text)

        # Recall memories
        memories = []
        if self._config.memory.top_k > 0:
            memories = self._memory.recall(embedding=query_emb, ctx=ctx, top_k=self._config.memory.top_k)

        # Select top-N tools via vector search (tools are stored as vectors)
        tools = await self._select_tools(query_embedding=query_emb)
        tool_block = tool_prompt_block(tools)

        system = _system_prompt(tool_block=tool_block, memories=memories)

        messages: list[dict[str, str]] = []
        # Include short-term buffer history to improve multi-turn.
        for turn in self._buffer.read(ctx=ctx):
            messages.append(turn)
        messages.append({"role": "user", "content": text})

        async with aiohttp.ClientSession() as session:
            tool_trace: list[dict[str, Any]] = []
            for step in range(self._config.agent.max_steps):
                action = await self._llm.next_action(system=system, messages=messages)

                kind = str(action.get("action", "")).strip().lower()
                if kind == "final":
                    final_text = str(action.get("final_text", "")).strip()
                    if not final_text:
                        raise UserVisibleError("Model returned action=final but final_text is empty")
                    cont = bool(action.get("continue_conversation", False))

                    # Update buffer
                    self._buffer.append(ctx=ctx, role="user", content=text)
                    self._buffer.append(ctx=ctx, role="assistant", content=final_text)

                    # Summarize buffer into memory when threshold hit
                    await self._maybe_summarize_into_memory(ctx=ctx)

                    return AgentResult(text=final_text, continue_conversation=cont, conversation_id=ctx.conversation_id)

                if kind == "tool":
                    tool_name = str(action.get("tool_name", "")).strip()
                    tool_args = action.get("tool_args")
                    if not tool_name or not isinstance(tool_args, dict):
                        raise UserVisibleError("Model returned action=tool but tool_name/tool_args missing")
                    tool = self._tools.get(tool_name)
                    if tool is None:
                        raise UserVisibleError(f"Model requested unknown tool: {tool_name}")

                    started = time.time()
                    result = await execute_tool(
                        tool=tool,
                        args=tool_args,
                        allow_python=self._config.tools.allow_python,
                        session=session,
                    )
                    duration_ms = int((time.time() - started) * 1000)

                    tool_trace.append({"tool": tool_name, "args": tool_args, "result": result, "ms": duration_ms})

                    messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                "TOOL_RESULT "
                                + json_dumps(
                                    {"tool": tool_name, "result": result, "duration_ms": duration_ms}
                                )
                            ),
                        }
                    )
                    continue

                raise UserVisibleError(f"Model returned unknown action: {kind}")

        raise UserVisibleError("Max agent steps reached without a final answer")

    async def _select_tools(self, *, query_embedding: list[float]) -> list[ToolSpec]:
        # Pull top candidates by embedding similarity.
        hits = self._store.query(kind="tool", query_embedding=query_embedding, top_k=self._config.rag.tools_top_n)
        tools: list[ToolSpec] = []
        for item, score in hits:
            name = item.metadata.get("tool_name")
            if not isinstance(name, str):
                continue
            tool = self._tools.get(name)
            if tool is None:
                continue
            tools.append(tool)
            _LOGGER.debug("Tool candidate %s score=%.3f", name, score)
        return tools

    async def _maybe_summarize_into_memory(self, *, ctx: ConversationContext) -> None:
        turns = self._buffer.count_turns(ctx=ctx)
        if turns < (self._config.memory.buffer_max_turns * 2):
            return

        # Summarize the buffer using the same LLM
        history = self._buffer.read(ctx=ctx)
        transcript = "\n".join([f"{m['role']}: {m['content']}" for m in history])

        system = (
            "You summarize chat transcripts into compact long-term memory. "
            "Return STRICT JSON only.\n\n"
            "Schema: { \"summary\": string }\n\n"
            "Summary rules:\n"
            "- Capture stable preferences, facts, names, and recurring goals\n"
            "- Avoid transient numbers/timestamps unless crucial\n"
            "- 3-8 short bullet points in one string\n"
        )

        action = await self._summarizer_llm.next_action(
            system=system,
            messages=[{"role": "user", "content": transcript}],
        )
        summary = str(action.get("summary", "")).strip()
        if not summary:
            # If summarization failed, don't block user.
            _LOGGER.warning("Summarization produced empty summary")
            self._buffer.clear(ctx=ctx)
            return

        # Store as memory
        emb = await self._embeddings.embed(summary)
        expires_at = None
        if self._config.memory.expiry_days_default > 0:
            expires_at = time.time() + (self._config.memory.expiry_days_default * 86400)
        self._memory.remember(embedding=emb, text=summary, ctx=ctx, expires_at=expires_at)

        self._buffer.clear(ctx=ctx)


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
