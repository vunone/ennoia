"""Load, stratify, and identity-disambiguate the CUAD QA dataset.

CUAD (`theatticusproject/cuad-qa`) ships in SQuAD format with one row per
``(contract, clause category)`` pair. Every question says "parts (if any)
of *this contract* related to …" but the query text itself never identifies
which contract — when every contract in a multi-doc index is asked every
category, retrieval has nothing to filter on and ~52% of questions
legitimately expect ``NOT_FOUND``. That collapses the benchmark into noise.

This module restores the disambiguating context a real user would naturally
provide: the contract identity. Each question is prefixed with a
parties/date/type clause parsed from the CUAD filename and gold spans, e.g.

    "In the 2019 Development Agreement between Fuelcell Energy Inc and
     ExxonMobil, highlight the parts (if any) of this contract related to
     'Non-Compete' …"

Gold spans are preserved verbatim. The only change is the wrapper around
the question; "this contract" now has a clear antecedent. Disclosure of
this methodology is in ``docs/cookbook/cuad-benchmark.md``.

Per-contract curation (see ``POS_PER_CONTRACT`` / ``NEG_PER_CONTRACT`` in
``benchmark.config``): we keep a fixed mix of positive (clause present) and
negative (clause absent) questions per contract so the benchmark measures
both retrieval quality and honest-``NOT_FOUND`` behaviour without being
dominated by the pre-rewrite 52% empty-gold trap.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, TypedDict, cast

from datasets import load_dataset  # type: ignore[import-untyped]

from benchmark.config import (
    DEFAULT_CONTRACT_COUNT,
    DEFAULT_QA_COUNT,
    HF_CACHE_DIR,
    HF_DATASET,
    NEG_PER_CONTRACT,
    POS_PER_CONTRACT,
    SAMPLE_PATH,
    SEED,
)


class Contract(TypedDict):
    source_id: str
    title: str
    text: str


class Question(TypedDict):
    question_id: str
    contract_id: str
    question: str
    category: str
    gold_answers: list[str]
    has_answer: bool


class Sample(TypedDict):
    contracts: list[Contract]
    questions: list[Question]
    seed: int
    contract_count: int
    qa_count: int


_CATEGORY_RE = re.compile(r'"([^"]+)"')


@dataclass(slots=True, frozen=True)
class _IdentityFacets:
    doc_type: str  # "Distributor Agreement", "SaaS Agreement", etc.
    primary_party: str  # Cleaned lead company name from the filename.
    second_party: str | None  # First additional party from CUAD gold, if clean.
    year: str | None  # Four-digit year parsed from filename or gold.
    governing_law: str | None  # Short phrase like "New York" or "Delaware".


def _extract_category(question: str) -> str:
    match = _CATEGORY_RE.search(question)
    return match.group(1) if match else "Unknown"


# Titles look like:
#   FuelcellEnergyInc_20191106_8-K_EX-10.1_11868007_EX-10.1_Development Agreement
#   MARTINMIDSTREAMPARTNERSLP_01_23_2004-EX-10.3-TRANSPORTATION SERVICES AGREEMENT
#   LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT
# Company is the first underscore/hyphen-separated token; year is the first
# 19xx/20xx run of 4 digits; doc_type is the trailing AGREEMENT-ending phrase.
_YEAR_RE = re.compile(r"(19|20)\d{2}")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_CORP_SUFFIX_RE = re.compile(
    r"(?i)(?<=\w)(?=(?:Inc|Corp|Corporation|Ltd|Limited|LLC|LLP|LP|Co|"
    r"Company|SA|NV|AG|PLC|GmbH|Holdings|Group)\b)"
)
_FORM_TOKEN_RE = re.compile(
    r"^(8-?K|10-?K|10-?Q|S-?\d+|F-?\d+|20-?F|EX-?\d+(?:\.\d+)?|\d+-?[A-Z]+)$",
    re.IGNORECASE,
)
# Match a trailing ``<modifier>... Agreement/Contract/...`` phrase anchored
# at end of string. Character class excludes comma so we stop at ", Dated ..."
# style annotations.
_DOC_TYPE_TAIL_RE = re.compile(
    r"(?i)"
    r"([A-Za-z][A-Za-z &/'-]{2,}?\s+"
    r"(?:agreement|contract|indenture|amendment|assignment|lease|license|"
    r"statement|addendum|schedule))\s*$"
)
_DOC_TYPE_TRIM_RE = re.compile(r"(?i)(?:,|\s+dated\b|\s+between\b|\s+effective\b).*$")


def _pretty_company(raw: str) -> str:
    """Turn ``FUELCELLENERGYINC`` / ``FuelcellEnergyInc`` / ``Lime Energy Co``
    into a readable phrase ("Fuelcell Energy Inc")."""
    cleaned = raw.replace(",", " ").replace(".", " ")
    # Split before any trailing corporate suffix ("Inc", "Corp", ...) even
    # when glued to the company name in all-caps or lowercase ("CARDAXINC"
    # -> "Cardax Inc", "Accurayinc" -> "Accuray Inc").
    with_suffix_split = _CORP_SUFFIX_RE.sub(" ", cleaned)
    # Split CamelCase boundaries next.
    # First split on whitespace (the suffix pass inserts a space), then on
    # CamelCase boundaries within each whitespace-delimited token.
    words: list[str] = []
    for token in with_suffix_split.split():
        words.extend(p for p in _CAMEL_SPLIT_RE.split(token) if p)
    out: list[str] = []
    for word in words:
        word = word.strip("_ -")
        if not word:
            continue
        # ALLCAPS / all-lowercase -> Capitalize (first letter up, rest down).
        # Already mixed-case (CamelCase words) stays as-is.
        if word.isupper() or word.islower():
            out.append(word.capitalize())
        else:
            out.append(word)
    result = " ".join(out).strip()
    return result or raw


def _pretty_doc_type(raw: str) -> str:
    """Normalise ``TRANSPORTATION SERVICES AGREEMENT, Dated ...`` to
    ``Transportation Services Agreement``."""
    trimmed = _DOC_TYPE_TRIM_RE.sub("", raw).strip()
    return " ".join(w.capitalize() for w in trimmed.split())


def _parse_title(title: str) -> tuple[str, str | None, str]:
    """Return (primary_party, year, doc_type) parsed from the CUAD filename."""
    # Tail match for doc type — everything up to and including the
    # "... Agreement/Contract/..." suffix.
    type_match = _DOC_TYPE_TAIL_RE.search(title)
    doc_type = _pretty_doc_type(type_match.group(1)) if type_match else "Agreement"

    year_match = _YEAR_RE.search(title)
    year = year_match.group(0) if year_match else None

    # Strip the trailing doc_type from the title, then take the first
    # segment that isn't a form code (``EX-10.1``, ``8-K``, etc.) and
    # isn't a pure date run.
    stem = title[: type_match.start()] if type_match else title
    tokens = re.split(r"[_\-\s]+", stem)
    primary_raw = ""
    for tok in tokens:
        if not tok:
            continue
        if _FORM_TOKEN_RE.match(tok):
            continue
        if tok.isdigit():
            continue
        primary_raw = tok
        break
    primary = _pretty_company(primary_raw) if primary_raw else "the contract party"
    return primary, year, doc_type


def _first_clean_gold(rows: list[dict[str, Any]], min_len: int = 4) -> str | None:
    """Return the first gold span that looks like a human-readable phrase.

    CUAD gold for ``Parties`` sometimes contains bare abbreviations
    ("FCE", "the Distributor") — those hurt identity context rather than
    helping, so we require a minimum length + at least one space for the
    ``second_party`` facet.
    """
    for row in rows:
        for text in row["answers"]["text"]:
            t = text.strip().strip(",;.:\"'")
            if len(t) >= min_len and " " in t and t.lower() not in {"the distributor"}:
                return t
    return None


_LAW_PATTERNS = (
    r"(?i)state of\s+([A-Za-z][A-Za-z ]*)",
    r"(?i)commonwealth of\s+([A-Za-z][A-Za-z ]*)",
    r"(?i)laws of\s+(?:the\s+)?(?:state of\s+)?([A-Za-z][A-Za-z ]*)",
    r"(?i)governed by\s+(?:and\s+construed\s+in\s+accordance\s+with\s+)?"
    r"(?:the\s+laws?\s+of\s+)?([A-Za-z][A-Za-z ]*?)\s+laws?\b",
    r"(?i)accordance with\s+(?:the\s+laws?\s+of\s+)?([A-Za-z][A-Za-z ]*?)\s+laws?\b",
)


def _short_law(raw: str | None) -> str | None:
    if raw is None:
        return None
    for pat in _LAW_PATTERNS:
        m = re.search(pat, raw)
        if not m:
            continue
        captured = m.group(1).strip()
        # Strip leading filler words that leaked into the capture, including
        # adjectives like "internal"/"substantive"/"applicable" that precede
        # the real jurisdiction name in phrases like "the internal laws of X".
        fillers = (
            "the ",
            "and ",
            "by ",
            "in ",
            "with ",
            "internal ",
            "substantive ",
            "applicable ",
            "domestic ",
            "federal ",
            "local ",
        )
        changed = True
        while changed:
            changed = False
            for filler in fillers:
                if captured.lower().startswith(filler):
                    captured = captured[len(filler) :]
                    changed = True
        if len(captured) > 40 or any(
            tok in captured.lower().split()
            for tok in ("shall", "agreement", "construed", "interpreted")
        ):
            continue
        words = [w.capitalize() if len(w) > 2 else w.lower() for w in captured.split()]
        result = " ".join(words).strip()
        stop = {
            "the",
            "a",
            "an",
            "this",
            "our",
            "any",
            "such",
            "internal",
            "substantive",
            "applicable",
            "domestic",
            "federal",
            "local",
        }
        if result and result.lower() not in stop and len(result) >= 3:
            return result
    return None


_NORM_COMPANY_RE = re.compile(r"[^a-z]")


def _same_company(a: str, b: str) -> bool:
    """Fuzzy check: do two party strings refer to the same entity?

    Compares alphabetic-only normalised prefixes so that
    ``"Rockymountainchocolatefactory"`` and
    ``"Rocky Mountain Chocolate Factory, Inc."`` match.
    """
    norm_a = _NORM_COMPANY_RE.sub("", a.lower())
    norm_b = _NORM_COMPANY_RE.sub("", b.lower())
    if not norm_a or not norm_b:
        return False
    prefix = 8
    return norm_a[:prefix] in norm_b or norm_b[:prefix] in norm_a


_PARTY_JUNK_RE = re.compile(
    r"(?i)\b(?:referred\s+to|hereinafter|collectively|individually|together\s+with|"
    r"shall\s+be|party|parties|herein)\b"
)


def _best_party_gold(parties_rows: list[dict[str, Any]], exclude: str) -> str | None:
    """Return the shortest clean CUAD 'Parties' gold span that looks like
    a company legal name (not a definitional clause) and is not the same
    company as ``exclude``."""
    candidates: list[str] = []
    seen: set[str] = set()
    for row in parties_rows:
        for text in row["answers"]["text"]:
            t = text.strip().strip(",;.:\"'")
            if not (4 <= len(t) <= 80 and " " in t):
                continue
            if _PARTY_JUNK_RE.search(t):
                continue
            if t not in seen:
                seen.add(t)
                candidates.append(t)
    # Prefer shortest clean name — CUAD party gold sometimes spans multiple
    # sentences; shorter = more likely a pure company legal name.
    candidates.sort(key=len)
    for cand in candidates:
        if not _same_company(cand, exclude):
            return cand
    return None


def _build_identity(
    title: str, rows_by_category: dict[str, list[dict[str, Any]]]
) -> _IdentityFacets:
    primary, year_from_title, doc_type = _parse_title(title)
    # Prefer an explicit Document Name gold span for the doc type when it
    # looks like a real title (short, no verbs). Otherwise stick with the
    # filename-parsed doc_type.
    doc_gold = _first_clean_gold(rows_by_category.get("Document Name", []))
    if (
        doc_gold
        and len(doc_gold) <= 60
        and "shall" not in doc_gold.lower()
        and "agreement" in doc_gold.lower()
    ):
        doc_type = _pretty_doc_type(doc_gold)
    second_party = _best_party_gold(rows_by_category.get("Parties", []), exclude=primary)
    # Gold-extracted year takes precedence over filename year when available.
    year = year_from_title
    for cat in ("Agreement Date", "Effective Date"):
        gold = _first_clean_gold(rows_by_category.get(cat, []), min_len=4)
        if gold:
            m = _YEAR_RE.search(gold)
            if m:
                year = m.group(0)
                break
    law_gold = _first_clean_gold(rows_by_category.get("Governing Law", []))
    governing_law = _short_law(law_gold)
    return _IdentityFacets(
        doc_type=doc_type,
        primary_party=primary,
        second_party=second_party,
        year=year,
        governing_law=governing_law,
    )


def _template_party(ident: _IdentityFacets, original: str) -> str:
    if ident.second_party:
        parties = f"{ident.primary_party} and {ident.second_party}"
    else:
        parties = ident.primary_party
    year_phrase = f" dated {ident.year}" if ident.year else ""
    return f"In the {ident.doc_type} between {parties}{year_phrase}, {original}"


def _template_title(ident: _IdentityFacets, original: str) -> str:
    year_phrase = f"{ident.year} " if ident.year else ""
    return f"For the {year_phrase}{ident.doc_type} involving {ident.primary_party}, {original}"


def _template_law(ident: _IdentityFacets, original: str) -> str:
    law_phrase = f" governed by {ident.governing_law} law" if ident.governing_law else ""
    return f"In the {ident.doc_type}{law_phrase} signed by {ident.primary_party}, {original}"


_TEMPLATES = (_template_party, _template_title, _template_law)


def _reformulate(question_id: str, ident: _IdentityFacets, original: str) -> str:
    """Deterministically choose a template by hashing the question id."""
    idx = abs(hash(question_id)) % len(_TEMPLATES)
    # Lower-case the first letter of the original so the prefix flows
    # grammatically ("..., highlight the parts..." not "..., Highlight...").
    if original and original[0].isupper():
        original = original[0].lower() + original[1:]
    return _TEMPLATES[idx](ident, original)


def _merge_rows_by_category(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cat[_extract_category(row["question"])].append(row)
    return by_cat


def _collapse_category(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    """Merge every row for one (contract, category) into one gold list + a
    representative row (for question_id, question text, context)."""
    seen: set[str] = set()
    merged: list[str] = []
    for row in rows:
        for text in row["answers"]["text"]:
            t = text.strip()
            if t and t not in seen:
                seen.add(t)
                merged.append(t)
    # Pick the first row as representative; CUAD's multiple rows per category
    # differ only in their extracted gold-span text, not the question.
    return merged, rows[0]


def _pick_pos_neg(
    rng: random.Random,
    by_cat: dict[str, list[dict[str, Any]]],
    pos_n: int,
    neg_n: int,
) -> tuple[
    list[tuple[str, list[str], dict[str, Any]]], list[tuple[str, list[str], dict[str, Any]]]
]:
    """Return (positives, negatives) as lists of (category, gold, rep-row)."""
    positives: list[tuple[str, list[str], dict[str, Any]]] = []
    negatives: list[tuple[str, list[str], dict[str, Any]]] = []
    # Skip the identity categories — they leak the answer the reformulated
    # query already contains.
    identity_categories = {
        "Document Name",
        "Parties",
        "Agreement Date",
        "Effective Date",
        "Governing Law",
    }
    for cat, rows in by_cat.items():
        if cat in identity_categories:
            continue
        gold, rep = _collapse_category(rows)
        if gold:
            positives.append((cat, gold, rep))
        else:
            negatives.append((cat, gold, rep))
    rng.shuffle(positives)
    rng.shuffle(negatives)
    return positives[:pos_n], negatives[:neg_n]


def build_sample(
    contract_count: int = DEFAULT_CONTRACT_COUNT,
    qa_count: int = DEFAULT_QA_COUNT,
    seed: int = SEED,
    pos_per_contract: int = POS_PER_CONTRACT,
    neg_per_contract: int = NEG_PER_CONTRACT,
) -> Sample:
    """Download CUAD (cached), curate a per-contract positive/negative mix,
    and reformulate every question with identity-disambiguating context."""
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(
        HF_DATASET,
        cache_dir=str(HF_CACHE_DIR),
        split="train",
        trust_remote_code=True,
    )

    rows_by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    contract_text: dict[str, str] = {}
    for row in raw:
        row_d = cast(dict[str, Any], row)
        title = cast(str, row_d["title"])
        rows_by_contract[title].append(row_d)
        contract_text.setdefault(title, cast(str, row_d["context"]))

    rng = random.Random(seed)
    all_contract_ids = sorted(rows_by_contract)
    rng.shuffle(all_contract_ids)

    contracts: list[Contract] = []
    questions: list[Question] = []
    per_contract = pos_per_contract + neg_per_contract

    for cid in all_contract_ids:
        if len(contracts) >= contract_count:
            break
        if len(questions) + per_contract > qa_count:
            break
        by_cat = _merge_rows_by_category(rows_by_contract[cid])
        # Skip contracts where we can't build identity — no point including
        # a query with no disambiguating context.
        ident = _build_identity(cid, by_cat)
        if ident.primary_party == "the contract party":
            continue
        positives, negatives = _pick_pos_neg(rng, by_cat, pos_per_contract, neg_per_contract)
        if len(positives) < pos_per_contract or len(negatives) < neg_per_contract:
            # Not enough material — skip rather than pad with unbalanced picks.
            continue

        contracts.append({"source_id": cid, "title": cid, "text": contract_text[cid]})
        for cat, gold, rep in positives + negatives:
            qid = cast(str, rep["id"])
            original_question = cast(str, rep["question"])
            questions.append(
                {
                    "question_id": qid,
                    "contract_id": cid,
                    "question": _reformulate(qid, ident, original_question),
                    "category": cat,
                    "gold_answers": gold,
                    "has_answer": bool(gold),
                }
            )

    return {
        "contracts": contracts,
        "questions": questions,
        "seed": seed,
        "contract_count": len(contracts),
        "qa_count": len(questions),
    }


def load_or_build_sample(
    contract_count: int = DEFAULT_CONTRACT_COUNT,
    qa_count: int = DEFAULT_QA_COUNT,
    seed: int = SEED,
    rebuild: bool = False,
) -> Sample:
    """Read ``sample.json`` if present (and matching), else regenerate it."""
    if not rebuild and SAMPLE_PATH.exists():
        with SAMPLE_PATH.open("r", encoding="utf-8") as fh:
            cached = cast(Sample, json.load(fh))
        if (
            cached.get("seed") == seed
            and cached.get("contract_count") == contract_count
            and cached.get("qa_count") <= qa_count
        ):
            return cached

    sample = build_sample(contract_count, qa_count, seed)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SAMPLE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(sample, fh, ensure_ascii=False, indent=2)
    return sample


if __name__ == "__main__":
    sample = load_or_build_sample(rebuild=True)
    print(
        f"contracts={sample['contract_count']} questions={sample['qa_count']} "
        f"seed={sample['seed']} -> {SAMPLE_PATH}"
    )
