from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from homeassistant.core import HomeAssistant


@dataclass(frozen=True)
class VectorItem:
    item_id: str
    kind: str  # "tool" | "memory"
    text: str
    embedding: list[float]
    metadata: dict[str, Any]
    created_at: float
    expires_at: float | None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for av, bv in zip(a, b, strict=True):
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na <= 0.0 or nb <= 0.0:
        return -1.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))


class VectorStoreSQLite:
    def __init__(self, *, hass: HomeAssistant, db_path: str) -> None:
        self._hass = hass
        self._db_path = db_path

    async def async_setup(self) -> None:
        await self._hass.async_add_executor_job(self._ensure_schema)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                  item_id TEXT PRIMARY KEY,
                  kind TEXT NOT NULL,
                  text TEXT NOT NULL,
                  embedding_json TEXT NOT NULL,
                  metadata_json TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  expires_at REAL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_kind ON vectors(kind)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_expires ON vectors(expires_at)")

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

    async def prune_expired(self) -> int:
        return await self._hass.async_add_executor_job(self._prune_expired)

    def _prune_expired(self) -> int:
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM vectors WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            return int(cur.rowcount)

    async def upsert(
        self,
        *,
        item_id: str,
        kind: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        expires_at: float | None = None,
    ) -> None:
        await self._hass.async_add_executor_job(
            self._upsert, item_id, kind, text, embedding, metadata or {}, expires_at
        )

    def _upsert(
        self,
        item_id: str,
        kind: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any],
        expires_at: float | None,
    ) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO vectors(item_id, kind, text, embedding_json, metadata_json, created_at, expires_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    kind=excluded.kind,
                    text=excluded.text,
                    embedding_json=excluded.embedding_json,
                    metadata_json=excluded.metadata_json,
                    expires_at=excluded.expires_at
                """,
                (
                    item_id,
                    kind,
                    text,
                    json.dumps(embedding, ensure_ascii=False),
                    json.dumps(metadata, ensure_ascii=False),
                    now,
                    expires_at,
                ),
            )

    async def delete_prefix(self, *, kind: str, prefix: str) -> int:
        return await self._hass.async_add_executor_job(self._delete_prefix, kind, prefix)

    def _delete_prefix(self, kind: str, prefix: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM vectors WHERE kind = ? AND item_id LIKE ?",
                (kind, f"{prefix}%"),
            )
            return int(cur.rowcount)

    async def query(
        self,
        *,
        kind: str,
        query_embedding: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[VectorItem, float]]:
        return await self._hass.async_add_executor_job(
            self._query, kind, query_embedding, top_k, metadata_filter
        )

    def _query(
        self,
        kind: str,
        query_embedding: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[tuple[VectorItem, float]]:
        results: list[tuple[VectorItem, float]] = []
        with self._connect() as conn:
            now = time.time()
            rows = conn.execute(
                """
                SELECT item_id, kind, text, embedding_json, metadata_json, created_at, expires_at
                FROM vectors
                WHERE kind = ? AND (expires_at IS NULL OR expires_at > ?)
                """,
                (kind, now),
            ).fetchall()

        for row in rows:
            metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            if metadata_filter:
                if any(metadata.get(k) != v for k, v in metadata_filter.items()):
                    continue

            emb = json.loads(row["embedding_json"])
            score = _cosine_similarity(query_embedding, emb)
            item = VectorItem(
                item_id=str(row["item_id"]),
                kind=str(row["kind"]),
                text=str(row["text"]),
                embedding=emb,
                metadata=metadata,
                created_at=float(row["created_at"]),
                expires_at=(float(row["expires_at"]) if row["expires_at"] is not None else None),
            )
            results.append((item, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # Buffer store helpers (short-term conversation memory)
    async def buffer_append(self, *, buffer_key: str, role: str, content: str) -> None:
        await self._hass.async_add_executor_job(self._buffer_append, buffer_key, role, content)

    def _buffer_append(self, buffer_key: str, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO buffer_turns(buffer_key, ts, role, content) VALUES(?, ?, ?, ?)",
                (buffer_key, time.time(), role, content),
            )

    async def buffer_read(self, *, buffer_key: str) -> list[dict[str, str]]:
        return await self._hass.async_add_executor_job(self._buffer_read, buffer_key)

    def _buffer_read(self, buffer_key: str) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM buffer_turns WHERE buffer_key = ? ORDER BY ts ASC",
                (buffer_key,),
            ).fetchall()
        return [{"role": str(r["role"]), "content": str(r["content"])} for r in rows]

    async def buffer_count(self, *, buffer_key: str) -> int:
        return await self._hass.async_add_executor_job(self._buffer_count, buffer_key)

    def _buffer_count(self, buffer_key: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM buffer_turns WHERE buffer_key = ?",
                (buffer_key,),
            ).fetchone()
        return int(row["c"]) if row else 0

    async def buffer_clear(self, *, buffer_key: str) -> int:
        return await self._hass.async_add_executor_job(self._buffer_clear, buffer_key)

    def _buffer_clear(self, buffer_key: str) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM buffer_turns WHERE buffer_key = ?", (buffer_key,))
            return int(cur.rowcount)
