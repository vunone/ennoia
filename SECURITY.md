# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.3.x   | Yes                |
| 0.2.x   | No (superseded)    |
| < 0.2   | No                 |

## Reporting a Vulnerability

Please report security issues privately by emailing `security@ennoia.dev`.
Do not open public GitHub issues for vulnerabilities.

We commit to:

- Acknowledging your report within 48 hours.
- Providing a status update within 7 days with an expected fix timeline.
- Crediting you in the CVE / release notes (opt-in) once a fix ships.

## Scope

**In scope**

- The `ennoia` package as published on PyPI (core, CLI, servers, stores,
  adapters, testing utilities).
- Filter-validation bypasses that could cause unauthorized data
  disclosure through the REST or MCP server.
- Auth bypasses in `ennoia.server.auth` or the REST/MCP request path.

**Out of scope**

- User-authored schemas in `examples/` — they are illustrative.
- Third-party adapters not shipped in this repository.
- Vulnerabilities in upstream dependencies (`pydantic`, `fastapi`,
  `fastmcp`, `qdrant-client`, `asyncpg`, `pgvector`, `ollama`, `openai`,
  `anthropic`, `sentence-transformers`). Report those to the upstream
  project; we will track and pin affected versions as needed.
