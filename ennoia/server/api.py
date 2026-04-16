"""FastAPI REST surface: ``/discover``, ``/filter``, ``/search``, ``/retrieve``, ``/index``, ``/delete``.

Every route reads from a :class:`~ennoia.server.context.ServerContext` built
once at application startup and delegates to the underlying
:class:`~ennoia.index.pipeline.Pipeline`. The route shapes mirror
``.ref/USAGE.md §5`` (agent flow) and ``.ref/FILTER_SPECS.md §Interface Consistency``
so the REST payload is identical to the MCP / CLI / SDK payload.

Filter-validation failures are surfaced as HTTP 422 with the error body
described in ``docs/filters.md §Filter Validation``.
"""

# pyright: reportUnusedFunction=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportInvalidTypeForm=false
# Route functions are registered via FastAPI decorators and fastapi's runtime
# dependency-injection layer isn't typed end-to-end; strict mode noise is
# scoped to this file rather than polluting the rest of the codebase.
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ennoia.index.exceptions import FilterValidationError
from ennoia.schema import describe
from ennoia.utils.imports import require_module

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import FastAPI

    from ennoia.server.context import ServerContext

__all__ = ["create_app"]


def create_app(ctx: ServerContext) -> FastAPI:
    """Build the FastAPI app bound to ``ctx``."""
    fastapi = require_module("fastapi", "server")
    FastAPI = fastapi.FastAPI
    HTTPException = fastapi.HTTPException
    Depends = fastapi.Depends
    Request = fastapi.Request
    fastapi_security = require_module("fastapi.security", "server")
    HTTPBearer = fastapi_security.HTTPBearer
    HTTPAuthorizationCredentials = fastapi_security.HTTPAuthorizationCredentials

    app = FastAPI(title="Ennoia", version=_version())

    bearer = HTTPBearer(auto_error=False)

    async def require_auth(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> None:
        token = credentials.credentials if credentials is not None else None
        if not await ctx.auth(token):
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.exception_handler(FilterValidationError)
    async def _filter_error(_request: Request, exc: FilterValidationError) -> Any:
        return fastapi.responses.JSONResponse(status_code=422, content=exc.to_dict())

    @app.get("/discover")
    async def discover(_: None = Depends(require_auth)) -> dict[str, Any]:
        return describe(ctx.pipeline.schemas())

    @app.post("/filter")
    async def filter_route(
        body: dict[str, Any], _: None = Depends(require_auth)
    ) -> dict[str, list[str]]:
        try:
            filters = _extract_filters(body)
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        ids = await ctx.pipeline.afilter(filters)
        return {"ids": ids}

    @app.post("/search")
    async def search(body: dict[str, Any], _: None = Depends(require_auth)) -> dict[str, Any]:
        query = body.get("query")
        if not isinstance(query, str):
            raise HTTPException(status_code=422, detail="'query' must be a string")
        filters = body.get("filters") or None
        filter_ids = body.get("filter_ids")
        index = body.get("index")
        top_k = int(body.get("top_k", 10))
        result = await ctx.pipeline.asearch(
            query=query,
            filters=filters,
            filter_ids=filter_ids,
            index=index,
            top_k=top_k,
        )
        return {
            "hits": [
                {
                    "source_id": hit.source_id,
                    "score": hit.score,
                    "structural": hit.structural,
                    "semantic": hit.semantic,
                }
                for hit in result.hits
            ]
        }

    @app.get("/retrieve/{source_id}")
    async def retrieve(source_id: str, _: None = Depends(require_auth)) -> dict[str, Any]:
        record = await ctx.pipeline.aretrieve(source_id)
        if record is None:
            raise HTTPException(status_code=404, detail="not_found")
        return record

    @app.post("/index")
    async def index_route(body: dict[str, Any], _: None = Depends(require_auth)) -> dict[str, Any]:
        text = body.get("text")
        source_id = body.get("source_id")
        if not isinstance(text, str) or not isinstance(source_id, str):
            raise HTTPException(status_code=422, detail="'text' and 'source_id' must be strings")
        result = await ctx.pipeline.aindex(text=text, source_id=source_id)
        return {
            "source_id": result.source_id,
            "rejected": result.rejected,
            "schemas_extracted": [*result.structural.keys(), *result.semantic.keys()],
        }

    @app.delete("/delete/{source_id}")
    async def delete_route(source_id: str, _: None = Depends(require_auth)) -> dict[str, bool]:
        removed = await ctx.pipeline.adelete(source_id)
        return {"removed": removed}

    return app


def _extract_filters(body: dict[str, Any]) -> dict[str, Any]:
    if "filters" in body and body["filters"] is not None:
        filters = body["filters"]
        if not isinstance(filters, dict):
            raise ValueError("'filters' must be an object")
        return dict(filters)  # type: ignore[arg-type]
    # Allow agents to POST the filter dict directly at the top level — the MCP
    # tool signature reads filters as kwargs, so REST is friendlier when we
    # accept both shapes.
    return {k: v for k, v in body.items() if k != "filters"}


def _version() -> str:
    from ennoia import __version__

    return __version__
