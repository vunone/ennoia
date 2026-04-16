"""PostgreSQL + pgvector hybrid store.

Schema (one table per ``collection``)::

    CREATE TABLE {collection} (
        source_id  TEXT PRIMARY KEY,
        data       JSONB NOT NULL,
        {index_a}_vector vector(N),
        {index_b}_vector vector(N),
        ...
    );

Vector columns materialise on first sight of a new semantic index and stay
nullable so documents that don't activate an index don't need an explicit
zero fill. Dimension is discovered from the first vector written.

Multiple pipelines can coexist in one database by passing distinct
``collection=`` names — each writes to its own table.

Filter translation is delegated to
:func:`ennoia.store.hybrid._sql_filter.build_where`, which emits parameterised
asyncpg ``$N`` placeholders. All 11 filter operators translate natively;
there is no Python post-filter residual for this backend.

Requires the ``pgvector`` extra (``asyncpg`` + ``pgvector``).
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnnecessaryIsInstance=false
# asyncpg ships without type stubs; relaxing strict-mode checks for the
# adapter surface keeps the rest of the codebase at pyright-strict while
# acknowledging the missing upstream annotations.
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from ennoia.store.base import HybridStore, validate_collection_name
from ennoia.store.hybrid._sql_filter import build_where
from ennoia.utils.imports import require_module

if TYPE_CHECKING:  # pragma: no cover
    import asyncpg


__all__ = ["PgVectorHybridStore"]


_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


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
        self._known_indices: set[str] = set()
        self._initialised = False

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

    async def _ensure_schema(self, vector_dims: dict[str, int]) -> None:
        conn = await self._get_conn()
        if not self._initialised:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            vec_cols = ",\n    ".join(
                f'"{_validate_index(name)}_vector" vector({dim})'
                for name, dim in vector_dims.items()
            )
            cols_sql = f",\n    {vec_cols}" if vec_cols else ""
            await conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} ("
                f"\n    source_id TEXT PRIMARY KEY,"
                f"\n    data JSONB NOT NULL{cols_sql}\n)"
            )
            self._known_indices.update(vector_dims.keys())
            self._initialised = True
            return

        # Table exists; add any new index columns.
        new = set(vector_dims.keys()) - self._known_indices
        for name in new:
            dim = vector_dims[name]
            await conn.execute(
                f"ALTER TABLE {self._table} ADD COLUMN IF NOT EXISTS "
                f'"{_validate_index(name)}_vector" vector({dim})'
            )
        self._known_indices.update(new)

    async def upsert(
        self,
        source_id: str,
        data: dict[str, Any],
        vectors: dict[str, list[float]],
    ) -> None:
        dims = {name: len(v) for name, v in vectors.items()}
        await self._ensure_schema(dims)
        conn = await self._get_conn()

        columns = ["source_id", "data"]
        placeholders: list[str] = ["$1", "$2"]
        params: list[Any] = [source_id, json.dumps(data)]
        updates = ["data = EXCLUDED.data"]
        idx = 3
        for name, vec in vectors.items():
            col = f'"{_validate_index(name)}_vector"'
            columns.append(col)
            placeholders.append(f"${idx}")
            params.append(_vector_literal(vec))
            updates.append(f"{col} = EXCLUDED.{col}")
            idx += 1

        await conn.execute(
            f"INSERT INTO {self._table} ({', '.join(columns)}) "
            f"VALUES ({', '.join(placeholders)}) "
            f"ON CONFLICT (source_id) DO UPDATE SET {', '.join(updates)}",
            *params,
        )

    async def hybrid_search(
        self,
        filters: dict[str, Any],
        query_vector: list[float],
        top_k: int,
        *,
        index: str | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        conn = await self._get_conn()
        # Default to the first-seen index when none is specified; for a
        # freshly-constructed store without prior upserts there is no index
        # and we return empty.
        using = index or (next(iter(sorted(self._known_indices))) if self._known_indices else None)
        if using is None:
            return []
        col = f'"{_validate_index(using)}_vector"'

        where_sql, where_params = build_where(filters)
        vector_param_index = len(where_params) + 1
        top_k_param_index = vector_param_index + 1
        sql = (
            f"SELECT source_id, data, ({col} <=> ${vector_param_index}) AS distance"
            f" FROM {self._table}"
            f" WHERE ({where_sql}) AND {col} IS NOT NULL"
            f" ORDER BY {col} <=> ${vector_param_index} ASC"
            f" LIMIT ${top_k_param_index}"
        )
        rows = await conn.fetch(sql, *where_params, _vector_literal(query_vector), top_k)
        out: list[tuple[str, float, dict[str, Any]]] = []
        for row in rows:
            payload = _json_load(row["data"])
            payload["source_id"] = row["source_id"]
            payload["index"] = using
            distance = float(row["distance"])
            # Convert pgvector cosine distance (0..2, lower is closer) to a
            # similarity score (higher is closer) so the pipeline's sort
            # direction matches every other backend.
            score = 1.0 - distance
            out.append((f"{row['source_id']}:{using}", score, payload))
        return out

    async def filter(self, filters: dict[str, Any]) -> list[str]:
        conn = await self._get_conn()
        where_sql, params = build_where(filters)
        rows = await conn.fetch(
            f"SELECT source_id FROM {self._table} WHERE {where_sql}",
            *params,
        )
        return [row["source_id"] for row in rows]

    async def get(self, source_id: str) -> dict[str, Any] | None:
        conn = await self._get_conn()
        row = await conn.fetchrow(
            f"SELECT data FROM {self._table} WHERE source_id = $1",
            source_id,
        )
        if row is None:
            return None
        return _json_load(row["data"])

    async def delete(self, source_id: str) -> bool:
        conn = await self._get_conn()
        result = await conn.execute(
            f"DELETE FROM {self._table} WHERE source_id = $1",
            source_id,
        )
        # asyncpg's ``execute`` returns a string like "DELETE 1" — parse the count.
        if isinstance(result, str) and result.startswith("DELETE "):
            return int(result.split()[1]) > 0
        return False  # pragma: no cover — every asyncpg DELETE returns this shape


def _validate_index(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise ValueError(
            f"Semantic index name {name!r} is not a safe SQL identifier. "
            "Ennoia index names are class names, which should always match; "
            "this error indicates a misconfigured schema."
        )
    return name


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
