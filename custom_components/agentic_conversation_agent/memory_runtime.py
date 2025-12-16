from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from .vector_store import VectorStoreSQLite


@dataclass(frozen=True)
class ConversationContext:
    speaker: str | None
    device_id: str | None
    user_id: str | None
    conversation_id: str | None
    language: str | None


def _scope_for(ctx: ConversationContext) -> str:
    if ctx.speaker:
        return f"speaker:{ctx.speaker}"
    if ctx.user_id:
        return f"user:{ctx.user_id}"
    if ctx.device_id:
        return f"device:{ctx.device_id}"
    return "global"


class MemoryManager:
    def __init__(self, *, store: VectorStoreSQLite) -> None:
        self._store = store

    async def prune(self) -> int:
        return await self._store.prune_expired()

    async def remember(
        self,
        *,
        embedding: list[float],
        text: str,
        ctx: ConversationContext,
        expires_at: float | None,
        replace_similar_threshold: float = 0.88,
    ) -> str:
        scope = _scope_for(ctx)
        existing = await self._store.query(
            kind="memory", query_embedding=embedding, top_k=1, metadata_filter={"scope": scope}
        )
        if existing and existing[0][1] >= replace_similar_threshold:
            item, _score = existing[0]
            await self._store.upsert(
                item_id=item.item_id,
                kind="memory",
                text=text,
                embedding=embedding,
                metadata={"scope": scope},
                expires_at=expires_at,
            )
            return item.item_id

        item_id = f"mem:{uuid.uuid4()}"
        await self._store.upsert(
            item_id=item_id,
            kind="memory",
            text=text,
            embedding=embedding,
            metadata={"scope": scope},
            expires_at=expires_at,
        )
        return item_id

    async def recall(self, *, embedding: list[float], ctx: ConversationContext, top_k: int) -> list[str]:
        scope = _scope_for(ctx)
        scoped = await self._store.query(
            kind="memory", query_embedding=embedding, top_k=top_k, metadata_filter={"scope": scope}
        )
        texts: list[str] = [item.text for item, _ in scoped]
        if len(texts) < top_k:
            global_hits = await self._store.query(
                kind="memory",
                query_embedding=embedding,
                top_k=top_k - len(texts),
                metadata_filter={"scope": "global"},
            )
            texts.extend([item.text for item, _ in global_hits])
        return texts


class BufferStore:
    def __init__(self, *, store: VectorStoreSQLite) -> None:
        self._store = store

    def _buffer_key(self, ctx: ConversationContext) -> str:
        if ctx.conversation_id:
            return f"cid:{ctx.conversation_id}"
        if ctx.speaker:
            return f"speaker:{ctx.speaker}"
        if ctx.user_id:
            return f"user:{ctx.user_id}"
        if ctx.device_id:
            return f"device:{ctx.device_id}"
        return "global"

    async def append(self, *, ctx: ConversationContext, role: str, content: str) -> None:
        await self._store.buffer_append(buffer_key=self._buffer_key(ctx), role=role, content=content)

    async def read(self, *, ctx: ConversationContext) -> list[dict[str, str]]:
        return await self._store.buffer_read(buffer_key=self._buffer_key(ctx))

    async def count_turns(self, *, ctx: ConversationContext) -> int:
        return await self._store.buffer_count(buffer_key=self._buffer_key(ctx))

    async def clear(self, *, ctx: ConversationContext) -> int:
        return await self._store.buffer_clear(buffer_key=self._buffer_key(ctx))


def expiry_timestamp(*, days: int) -> float | None:
    if days <= 0:
        return None
    return time.time() + (days * 86400)
