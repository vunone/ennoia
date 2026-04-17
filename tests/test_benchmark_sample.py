"""Unit tests for benchmark.data.loader — identity-aware reformulation.

We don't download CUAD in tests; we build a tiny in-memory row set that
mirrors CUAD's SQuAD shape and exercise every piece of the reformulation
pipeline: filename parsing, gold-based identity extraction, template
choice, and per-contract pos/neg selection.
"""

from __future__ import annotations

from typing import Any

from benchmark.data import loader
from benchmark.data.loader import (
    _build_identity,
    _parse_title,
    _pretty_company,
    _reformulate,
    _short_law,
    _template_law,
    _template_party,
    _template_title,
    build_sample,
)


def test_pretty_company_splits_corporate_suffix() -> None:
    assert _pretty_company("FuelcellEnergyInc") == "Fuelcell Energy Inc"
    assert _pretty_company("CORIOINC") == "Corio Inc"
    assert _pretty_company("Accurayinc") == "Accuray Inc"
    assert _pretty_company("Lime Energy Co") == "Lime Energy Co"


def test_parse_title_extracts_primary_year_doctype() -> None:
    primary, year, doc = _parse_title(
        "FuelcellEnergyInc_20191106_8-K_EX-10.1_11868007_EX-10.1_Development Agreement"
    )
    assert primary == "Fuelcell Energy Inc"
    assert year == "2019"
    assert doc == "Development Agreement"

    primary, year, doc = _parse_title(
        "MARTINMIDSTREAMPARTNERSLP_01_23_2004-EX-10.3-TRANSPORTATION SERVICES AGREEMENT"
    )
    assert primary.startswith("Martinmidstreampartners")  # ALLCAPS glue stays
    assert year == "2004"
    assert doc == "Transportation Services Agreement"


def test_short_law_extracts_jurisdiction() -> None:
    assert (
        _short_law("This Agreement shall be governed by the laws of the State of Delaware.")
        == "Delaware"
    )
    assert (
        _short_law("construed in accordance with New York law without regard to conflicts")
        == "New York"
    )
    assert _short_law("by the Commonwealth of Massachusetts") == "Massachusetts"
    # Junk gold -> None, not a broken leading stop-word.
    assert (
        _short_law("This Agreement shall be governed by and construed in accordance with law.")
        is None
    )


def _row(title: str, question: str, gold: list[str], qid: str) -> dict[str, Any]:
    return {
        "id": qid,
        "title": title,
        "context": f"Text of {title}",
        "question": question,
        "answers": {"text": gold, "answer_start": [0] * len(gold)},
    }


TITLE_A = "AcmeCorp_20190315_10-K_EX-10.1_Distribution Agreement"


def _by_cat(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    from collections import defaultdict

    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[loader._extract_category(r["question"])].append(r)
    return dict(out)


def test_build_identity_combines_filename_and_gold() -> None:
    rows = [
        _row(TITLE_A, 'Highlight the parts related to "Parties"...', ["Beta Industries Inc"], "q1"),
        _row(
            TITLE_A,
            'Highlight the parts related to "Governing Law"...',
            ["This Agreement shall be governed by the laws of the State of California"],
            "q2",
        ),
        _row(
            TITLE_A,
            'Highlight the parts related to "Document Name"...',
            ["Distribution Agreement"],
            "q3",
        ),
        _row(
            TITLE_A, 'Highlight the parts related to "Agreement Date"...', ["March 15, 2019"], "q4"
        ),
    ]
    ident = _build_identity(TITLE_A, _by_cat(rows))
    assert ident.primary_party == "Acme Corp"
    assert ident.second_party == "Beta Industries Inc"
    assert ident.year == "2019"
    assert ident.governing_law == "California"
    assert ident.doc_type == "Distribution Agreement"


def test_templates_include_identity_markers() -> None:
    ident = loader._IdentityFacets(
        doc_type="Distribution Agreement",
        primary_party="Acme Corp",
        second_party="Beta Industries Inc",
        year="2019",
        governing_law="Delaware",
    )
    orig = 'highlight the parts related to "Non-Compete"...'
    assert "Acme Corp" in _template_party(ident, orig)
    assert "Beta Industries Inc" in _template_party(ident, orig)
    assert "2019" in _template_title(ident, orig)
    assert "Delaware" in _template_law(ident, orig)
    assert "Distribution Agreement" in _template_law(ident, orig)


def test_reformulate_is_deterministic_per_qid() -> None:
    ident = loader._IdentityFacets("Agreement", "Acme", None, "2019", None)
    a = _reformulate("q-1", ident, "highlight clauses")
    b = _reformulate("q-1", ident, "highlight clauses")
    assert a == b


def test_build_sample_curates_pos_neg_split(monkeypatch: Any) -> None:
    # Construct a synthetic CUAD-shaped dataset with two contracts, each
    # having ten positive + three negative categories beyond the identity
    # questions, then verify the loader produces the configured mix.
    def make_contract(title: str, counterparty: str, doc_name_gold: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        rows.append(_row(title, 'parts related to "Parties"', [counterparty], f"{title}-parties"))
        rows.append(
            _row(
                title,
                'parts related to "Governing Law"',
                ["laws of the State of Delaware"],
                f"{title}-law",
            )
        )
        rows.append(
            _row(title, 'parts related to "Document Name"', [doc_name_gold], f"{title}-name")
        )
        rows.append(
            _row(title, 'parts related to "Agreement Date"', ["January 5, 2020"], f"{title}-date")
        )
        positives = [
            "Non-Compete",
            "Exclusivity",
            "Cap On Liability",
            "Audit Rights",
            "License Grant",
            "Insurance",
            "Warranty Duration",
            "Expiration Date",
            "Renewal Term",
            "Change Of Control",
        ]
        negatives = [
            "Source Code Escrow",
            "Uncapped Liability",
            "Most Favored Nation",
            "Liquidated Damages",
        ]
        for i, cat in enumerate(positives):
            rows.append(
                _row(
                    title,
                    f'Highlight the parts related to "{cat}" that should be reviewed.',
                    [f"gold span for {cat}"],
                    f"{title}-pos-{i}",
                )
            )
        for i, cat in enumerate(negatives):
            rows.append(
                _row(
                    title,
                    f'Highlight the parts related to "{cat}" that should be reviewed.',
                    [],
                    f"{title}-neg-{i}",
                )
            )
        return rows

    fake_raw: list[dict[str, Any]] = []
    fake_raw.extend(
        make_contract(
            "AcmeCorp_20200105_8-K_Distribution Agreement",
            "Beta Inc",
            "Distribution Agreement",
        )
    )
    fake_raw.extend(
        make_contract(
            "ZetaPlc_20210202_10-K_Service Agreement",
            "Omega Corp",
            "Service Agreement",
        )
    )

    monkeypatch.setattr(loader, "load_dataset", lambda *a, **kw: fake_raw)

    sample = build_sample(
        contract_count=2, qa_count=20, seed=7, pos_per_contract=5, neg_per_contract=2
    )
    assert sample["contract_count"] == 2
    # 2 contracts * (5 pos + 2 neg) = 14
    assert sample["qa_count"] == 14
    per_contract: dict[str, list[bool]] = {}
    for q in sample["questions"]:
        per_contract.setdefault(q["contract_id"], []).append(q["has_answer"])
    for cid, flags in per_contract.items():
        pos = sum(1 for f in flags if f)
        neg = sum(1 for f in flags if not f)
        assert pos == 5, f"{cid}: expected 5 positives, got {pos}"
        assert neg == 2, f"{cid}: expected 2 negatives, got {neg}"
    # Every reformulated query contains the primary party name + doc type.
    for q in sample["questions"]:
        if q["contract_id"].startswith("AcmeCorp"):
            assert "Acme Corp" in q["question"]
            assert "Distribution Agreement" in q["question"]
        else:
            assert "Zeta Plc" in q["question"]
            assert "Service Agreement" in q["question"]
    # Identity categories are excluded from the sampled questions.
    for q in sample["questions"]:
        assert q["category"] not in {
            "Parties",
            "Governing Law",
            "Document Name",
            "Agreement Date",
            "Effective Date",
        }
