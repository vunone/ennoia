"""PostgreSQL + pgvector hybrid store.

Schema (one table per ``collection``)::

    CREATE TABLE {collection} (
        vector_id   TEXT PRIMARY KEY,
        source_id   TEXT NOT NULL,
        index_name  TEXT NOT NULL,
        unique_key  TEXT,
        text        TEXT,
        data        JSONB NOT NULL,
        vector      vector(N)
    );

A document produces one row per :class:`~ennoia.store.base.VectorEntry`: one
for each ``BaseSemantic`` answer, N for each ``BaseCollection`` with N
entities. The full structural ``data`` payload is copied onto every row so a
single SQL query can filter on structural fields and rank by vector similarity
at the same time. The pipeline collapses rows back to one hit per ``source_id``
at the search boundary.

Dimension is discovered from the first vector written; the ``vector`` column
is added in place (``ALTER TABLE``) if the table was created before any
vectors existed. Multiple pipelines coexist in one database by passing
distinct ``collection=`` names — each writes to its own table.

Filter translation is delegated to
:func:`ennoia.store.hybrid._sql_filter.build_where`, which emits parameterised
asyncpg ``$N`` placeholders against the ``data`` jsonb column.

Requires the ``pgvector`` extra (``asyncpg`` + ``pgvector``).
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnnecessaryIsInstance=false
# asyncpg ships without type stubs; relaxing strict-mode checks for the
# adapter surface keeps the rest of the codebase at pyright-strict while
# acknowledging the missing upstream annotations.
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ennoia.store.base import HybridStore, VectorEntry, validate_collection_name
from ennoia.store.hybrid._sql_filter import build_where
from ennoia.utils.ids import make_semantic_vector_id
from ennoia.utils.imports import require_module

if TYPE_CHECKING:  # pragma: no cover
    import asyncpg


__all__ = ["PgVectorHybridStore"]


class PgVectorHybridStore(HybridStore):
    def __init__(
        self,
        dsn: str,
        *,
        collection: str = "documents",
        connection: asyncpg.Connection | None = None,
    ) -> None:
        self.dsn = dsn
        self.collection = validate_collection_name(collection)
        self._table = self.collection
        self._conn: asyncpg.Connection | None = connection
        self._table_created = False
        self._vector_dim: int | None = None

    async def _get_conn(self) -> asyncpg.Connection:
        conn = self._conn
        if conn is None:
            asyncpg = require_module("asyncpg", "pgvector")
            pgvector_mod = require_module("pgvector.asyncpg", "pgvector")
            conn = await asyncpg.connect(self.dsn)
            await pgvector_mod.register_vector(conn)
            self._conn = conn
        return conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def _ensure_schema(self, sample_dim: int | None) -> None:
        conn = await self._get_conn()
        if not self._table_created:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            vector_col = f", vector vector({sample_dim})" if sample_dim is not None else ""
            await conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} ("
                f"\n    vector_id   TEXT PRIMARY KEY,"
                f"\n    source_id   TEXT NOT NULL,"
                f"\n    index_name  TEXT NOT NULL,"
                f"\n    unique_key  TEXT,"
                f"\n    text        TEXT,"
                f"\n    data        JSONB NOT NULL"
                f"{vector_col}"
                f"\n)"
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_source_idx ON {self._table} (source_id)"
            )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_index_name_idx "
                f"ON {self._table} (index_name)"
            )
            self._table_created = True
            self._vector_dim = sample_dim
            return

        # Table exists. If the vector column wasn't created at CREATE TABLE
        # time (first upsert had no vectors) and we now see one, add it.
        if self._vector_dim is None and sample_dim is not None:
            await conn.execute(
                f"ALTER TABLE {self._table} ADD COLUMN IF NOT EXISTS vector vector({sample_dim})"
            )
            self._vector_dim = sample_dim

    async def upsert(
        self,
        source_id: str,
        data: dict[str, Any],
        entries: list[VectorEntry],
    ) -> None:
        # Replace-by-source semantics: drop any prior rows for this document
        # before inserting the new set. Simpler than row-level UPSERT and
        # correct for N → M cardinality changes (e.g., a collection shrinking
        # on re-index).
        sample_dim = len(entries[0].vector) if entries else None
        await self._ensure_schema(sample_dim)
        conn = await self._get_conn()
        data_json = json.dumps(data)

        async with conn.transaction():
            await conn.execute(
                f"DELETE FROM {self._table} WHERE source_id = $1",
                source_id,
            )
            if not entries:
                # No vectors to persist, but we still record structural state
                # so ``get()`` / ``filter()`` can see this document. Use a
                # stable reserved sentinel index name.
                await conn.execute(
                    f"INSERT INTO {self._table} "
                    f"(vector_id, source_id, index_name, unique_key, text, data) "
                    f"VALUES ($1, $2, $3, NULL, NULL, $4::jsonb)",
                    make_semantic_vector_id(source_id, ""),
                    source_id,
                    "",
                    data_json,
                )
                return

            for entry in entries:
                vector_id = make_semantic_vector_id(source_id, entry.index_name, entry.unique)
                await conn.execute(
                    f"INSERT INTO {self._table} "
                    f"(vector_id, source_id, index_name, unique_key, text, data, vector) "
                    f"VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::vector)",
                    vector_id,
                    source_id,
                    entry.index_name,
                    entry.unique,
                    entry.text,
                    data_json,
                    _vector_literal(entry.vector),
                )

    async def hybrid_search(
        self,
        filters: dict[str, Any],
        query_vector: list[float],
        top_k: int,
        *,
        index: str | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        if not self._table_created or self._vector_dim is None:
            return []

        conn = await self._get_conn()
        where_sql, where_params = build_where(filters)
        params: list[Any] = list(where_params)

        clauses = [f"({where_sql})", "vector IS NOT NULL"]
        if index is not None:
            params.append(index)
            clauses.append(f"index_name = ${len(params)}")

        params.append(_vector_literal(query_vector))
        vector_param_idx = len(params)
        params.append(top_k)
        top_k_param_idx = len(params)

        sql = (
            f"SELECT vector_id, source_id, index_name, unique_key, text, data,"
            f" (vector <=> ${vector_param_idx}::vector) AS distance"
            f" FROM {self._table}"
            f" WHERE {' AND '.join(clauses)}"
            f" ORDER BY vector <=> ${vector_param_idx}::vector ASC"
            f" LIMIT ${top_k_param_idx}"
        )
        rows = await conn.fetch(sql, *params)
        out: list[tuple[str, float, dict[str, Any]]] = []
        for row in rows:
            payload = _json_load(row["data"])
            payload["source_id"] = row["source_id"]
            payload["index"] = row["index_name"]
            payload["text"] = row["text"] or ""
            if row["unique_key"] is not None:
                payload["unique"] = row["unique_key"]
            # pgvector cosine distance (0..2, lower is closer) → similarity.
            score = 1.0 - float(row["distance"])
            out.append((row["vector_id"], score, payload))
        return out

    async def filter(self, filters: dict[str, Any]) -> list[str]:
        if not self._table_created:
            return []
        conn = await self._get_conn()
        where_sql, params = build_where(filters)
        rows = await conn.fetch(
            f"SELECT DISTINCT source_id FROM {self._table} WHERE {where_sql}",
            *params,
        )
        return [row["source_id"] for row in rows]

    async def get(self, source_id: str) -> dict[str, Any] | None:
        if not self._table_created:
            return None
        conn = await self._get_conn()
        row = await conn.fetchrow(
            f"SELECT data FROM {self._table} WHERE source_id = $1 LIMIT 1",
            source_id,
        )
        if row is None:
            return None
        return _json_load(row["data"])

    async def delete(self, source_id: str) -> bool:
        if not self._table_created:
            return False
        conn = await self._get_conn()
        result = await conn.execute(
            f"DELETE FROM {self._table} WHERE source_id = $1",
            source_id,
        )
        if isinstance(result, str) and result.startswith("DELETE "):
            return int(result.split()[1]) > 0
        return False  # pragma: no cover — every asyncpg DELETE returns this shape


def _vector_literal(vec: list[float]) -> str:
    # pgvector accepts ``[1.0, 2.0, ...]`` text literals and also native
    # ``numpy.ndarray`` via ``pgvector.asyncpg``'s codec. Stick to text for
    # backend independence — avoids a hard numpy dependency in the pg path.
    return "[" + ", ".join(repr(float(x)) for x in vec) + "]"


def _json_load(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)  # type: ignore[arg-type]
    if isinstance(value, str):
        loaded = json.loads(value)
        if isinstance(loaded, dict):
            return dict(loaded)  # type: ignore[arg-type]
    return {}  # pragma: no cover — asyncpg always returns dict for jsonb
