from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

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

    def prune(self) -> int:
        return self._store.prune_expired()

    def remember(
        self,
        *,
        embedding: list[float],
        text: str,
        ctx: ConversationContext,
        expires_at: float | None,
        replace_similar_threshold: float = 0.88,
    ) -> str:
        scope = _scope_for(ctx)

        # Dedupe: replace closest existing memory within same scope.
        existing = self._store.query(kind="memory", query_embedding=embedding, top_k=1, metadata_filter={"scope": scope})
        if existing and existing[0][1] >= replace_similar_threshold:
            item, _score = existing[0]
            self._store.upsert(
                item_id=item.item_id,
                kind="memory",
                text=text,
                embedding=embedding,
                metadata={"scope": scope},
                expires_at=expires_at,
            )
            return item.item_id

        item_id = f"mem:{uuid.uuid4()}"
        self._store.upsert(
            item_id=item_id,
            kind="memory",
            text=text,
            embedding=embedding,
            metadata={"scope": scope},
            expires_at=expires_at,
        )
        return item_id

    def recall(
        self,
        *,
        embedding: list[float],
        ctx: ConversationContext,
        top_k: int,
    ) -> list[str]:
        scope = _scope_for(ctx)
        # Query scope-specific first, then global.
        scoped = self._store.query(kind="memory", query_embedding=embedding, top_k=top_k, metadata_filter={"scope": scope})
        texts: list[str] = [item.text for item, _ in scoped]
        if len(texts) < top_k:
            global_hits = self._store.query(kind="memory", query_embedding=embedding, top_k=top_k - len(texts), metadata_filter={"scope": "global"})
            texts.extend([item.text for item, _ in global_hits])
        return texts


class BufferStore:
    def __init__(self, *, db_path: str) -> None:
        import sqlite3

        self._db_path = db_path
        self._ensure_schema()

    def _connect(self):
        import sqlite3

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS buffer_turns (
                  buffer_key TEXT NOT NULL,
                  ts REAL NOT NULL,
                  role TEXT NOT NULL,
                  content TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_buffer_key_ts ON buffer_turns(buffer_key, ts)")

    def _buffer_key(self, ctx: ConversationContext) -> str:
        # Keep short-term memory per conversation if available, else per speaker/user/device.
        if ctx.conversation_id:
            return f"cid:{ctx.conversation_id}"
        if ctx.speaker:
            return f"speaker:{ctx.speaker}"
        if ctx.user_id:
            return f"user:{ctx.user_id}"
        if ctx.device_id:
            return f"device:{ctx.device_id}"
        return "global"

    def append(self, *, ctx: ConversationContext, role: str, content: str) -> None:
        key = self._buffer_key(ctx)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO buffer_turns(buffer_key, ts, role, content) VALUES(?, ?, ?, ?)",
                (key, time.time(), role, content),
            )

    def read(self, *, ctx: ConversationContext) -> list[dict[str, str]]:
        key = self._buffer_key(ctx)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM buffer_turns WHERE buffer_key = ? ORDER BY ts ASC",
                (key,),
            ).fetchall()
        return [{"role": str(r["role"]), "content": str(r["content"])} for r in rows]

    def count_turns(self, *, ctx: ConversationContext) -> int:
        key = self._buffer_key(ctx)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM buffer_turns WHERE buffer_key = ?",
                (key,),
            ).fetchone()
        return int(row["c"]) if row else 0

    def clear(self, *, ctx: ConversationContext) -> int:
        key = self._buffer_key(ctx)
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM buffer_turns WHERE buffer_key = ?", (key,))
            return int(cur.rowcount)
