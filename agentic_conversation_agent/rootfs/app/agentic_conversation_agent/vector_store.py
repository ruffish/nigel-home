from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Iterable


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
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._ensure_schema()

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

    def prune_expired(self) -> int:
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM vectors WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            return int(cur.rowcount)

    def upsert(
        self,
        *,
        item_id: str,
        kind: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        expires_at: float | None = None,
    ) -> None:
        now = time.time()
        metadata = metadata or {}
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

    def delete_prefix(self, *, kind: str, prefix: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM vectors WHERE kind = ? AND item_id LIKE ?",
                (kind, f"{prefix}%"),
            )
            return int(cur.rowcount)

    def _iter_kind(self, kind: str) -> Iterable[sqlite3.Row]:
        with self._connect() as conn:
            now = time.time()
            cur = conn.execute(
                """
                SELECT item_id, kind, text, embedding_json, metadata_json, created_at, expires_at
                FROM vectors
                WHERE kind = ? AND (expires_at IS NULL OR expires_at > ?)
                """,
                (kind, now),
            )
            yield from cur.fetchall()

    def query(
        self,
        *,
        kind: str,
        query_embedding: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[VectorItem, float]]:
        results: list[tuple[VectorItem, float]] = []
        for row in self._iter_kind(kind):
            metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            if metadata_filter:
                ok = True
                for k, v in metadata_filter.items():
                    if metadata.get(k) != v:
                        ok = False
                        break
                if not ok:
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

    def get_by_id(self, item_id: str) -> VectorItem | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT item_id, kind, text, embedding_json, metadata_json, created_at, expires_at FROM vectors WHERE item_id = ?",
                (item_id,),
            ).fetchone()
            if not row:
                return None
            return VectorItem(
                item_id=str(row["item_id"]),
                kind=str(row["kind"]),
                text=str(row["text"]),
                embedding=json.loads(row["embedding_json"]),
                metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
                created_at=float(row["created_at"]),
                expires_at=(float(row["expires_at"]) if row["expires_at"] is not None else None),
            )
