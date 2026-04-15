#!/usr/bin/env bash
#
# End-to-end CLI walkthrough — mirrors `docs/cli.md`.
#
# Prerequisites:
#   pip install "ennoia[cli,ollama,sentence-transformers,filesystem]"
#   ollama pull qwen3:0.6b
#
# Run from the repository root:
#   ./examples/cli_walkthrough.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCHEMA="$ROOT/examples/schemas.py"
DOCS="$ROOT/examples/fixtures"
STORE="$ROOT/examples/tmp_index"

rm -rf "$STORE"

echo "=== ennoia try (single file) ==="
ennoia try "$DOCS/case_001.txt" --schema "$SCHEMA"

echo
echo "=== ennoia index (whole folder) ==="
ennoia index "$DOCS" \
    --schema "$SCHEMA" \
    --store "$STORE" \
    --llm ollama:qwen3:0.6b \
    --embedding sentence-transformers:all-MiniLM-L6-v2

echo
echo "=== ennoia search (legal topic, filter by category) ==="
ennoia search "employer duty to accommodate disability" \
    --schema "$SCHEMA" \
    --store "$STORE" \
    --filter "category=legal" \
    --top-k 3

echo
echo "=== ennoia search (invalid operator triggers filter validation) ==="
ennoia search "anything" \
    --schema "$SCHEMA" \
    --store "$STORE" \
    --filter "category__gt=legal" || echo "(expected validation failure)"
