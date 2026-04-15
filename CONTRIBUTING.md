# Contributing to Ennoia

Ennoia is an Apache-2.0 open-source project and welcomes contributions.
Start with an issue describing the change if it isn't obvious from an
existing one.

## Dev setup

```bash
git clone https://github.com/ennoia-ai/ennoia
cd ennoia
uv venv
source .venv/bin/activate
pip install -e ".[all,dev]"
pre-commit install
```

Everything is optional under an extra. The development install pulls in
every adapter + store backend so the full test suite runs.

## Quality gates

Four gates must be green before merging — CI enforces each.

```bash
ruff check .           # lint
ruff format --check .  # formatting
pyright                # strict type-check
pytest                 # tests
```

`ruff` and `pyright` are non-negotiable; unexplained `# pyright: ignore`
or `# noqa` comments are not accepted.

## Testing conventions

- Every new module ships with a test module under `tests/`.
- Tests use pytest, `pytest-asyncio` (auto mode), and hand-rolled fakes.
  No mocking library is required in the default test path — the
  pipeline's protocols make structural typing cheap.
- Optional-dependency tests guard with `pytest.importorskip(...)` so the
  default install can run a useful subset.
- Integration tests against live external services (Qdrant, OpenAI,
  Anthropic) are reserved for Stage 3.

## Pull request checklist

- [ ] Issue or PR description explains the *why*, not just the *what*.
- [ ] Public API changes are covered in `docs/` (not `.ref/`).
- [ ] `CHANGELOG.md` entry under `[Unreleased]`.
- [ ] Tests added or updated for every behavioral change.
- [ ] All four quality gates pass locally.

## Governance

- Breaking changes to the public SDK require a minor-version bump
  pre-1.0, a major bump post-1.0.
- Filter-language changes (operators, inference rules, error shape)
  touch every surface (SDK, CLI, MCP, REST) — expect review from
  multiple maintainers.
- Adapter additions ship as extras; the core package depends only on
  Pydantic.
