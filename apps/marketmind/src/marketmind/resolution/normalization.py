"""Name normalization for entity matching.

"Apple Inc.", "APPLE, INC", and "Apple Incorporated" are one company. Getting
that right is unglamorous string work, but it is the difference between a graph
with one Apple node and a graph with five — and five Apples make every
downstream traversal quietly wrong.
"""

from __future__ import annotations

import re
import unicodedata

#: Legal-form suffixes stripped before comparison. Order matters: multi-word
#: forms must be removed before their single-word constituents.
_LEGAL_SUFFIXES: tuple[str, ...] = (
    "public limited company",
    "limited liability company",
    "société anonyme",
    "aktiengesellschaft",
    "incorporated",
    "corporation",
    "company",
    "limited",
    "holdings",
    "holding",
    "group",
    "s.a.b. de c.v.",
    "s.a. de c.v.",
    "pte ltd",
    "pty ltd",
    "co ltd",
    "plc",
    "llc",
    "lllp",
    "llp",
    "lp",
    "ltd",
    "inc",
    "corp",
    "nv",
    "bv",
    "ag",
    "sa",
    "se",
    "as",
    "ab",
    "oy",
    "kk",
    "gmbh",
    "spa",
    "srl",
)

#: Honorifics and post-nominals removed from person names.
_PERSON_PREFIXES = frozenset({"mr", "mrs", "ms", "miss", "dr", "prof", "sir", "dame"})
_PERSON_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "phd", "md", "mba", "cfa", "cpa"})

_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")
_AMPERSAND = re.compile(r"\s*&\s*")


def strip_accents(value: str) -> str:
    """Fold accented characters to ASCII so "Nestlé" matches "Nestle"."""
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize_company_name(name: str) -> str:
    """Reduce a company name to a comparable canonical form.

    Repeatedly strips trailing legal suffixes, because real filings stack them
    ("Acme Holdings Group Ltd").
    """
    working = strip_accents(name).lower().strip()
    working = _AMPERSAND.sub(" and ", working)
    working = _PUNCTUATION.sub(" ", working)
    working = _WHITESPACE.sub(" ", working).strip()

    changed = True
    while changed:
        changed = False
        for suffix in _LEGAL_SUFFIXES:
            cleaned_suffix = _PUNCTUATION.sub(" ", suffix)
            cleaned_suffix = _WHITESPACE.sub(" ", cleaned_suffix).strip()
            if working.endswith(f" {cleaned_suffix}"):
                working = working[: -len(cleaned_suffix) - 1].strip()
                changed = True
                break

    return _WHITESPACE.sub(" ", working).strip()


def normalize_person_name(name: str) -> str:
    """Reduce a person's name to a comparable canonical form.

    Handles "Cook, Timothy D." as well as "Dr. Timothy D. Cook Jr.", and drops
    middle initials, which appear inconsistently across sources.
    """
    working = strip_accents(name).lower().strip()

    # "Last, First Middle" -> "First Middle Last"
    if working.count(",") == 1:
        last, _, first = working.partition(",")
        if first.strip() and last.strip():
            working = f"{first.strip()} {last.strip()}"

    working = _PUNCTUATION.sub(" ", working)
    tokens = [token for token in _WHITESPACE.split(working) if token]

    while tokens and tokens[0] in _PERSON_PREFIXES:
        tokens.pop(0)
    while tokens and tokens[-1] in _PERSON_SUFFIXES:
        tokens.pop()

    # Drop single-letter middle initials, keeping first and last names.
    if len(tokens) > 2:
        tokens = [tokens[0]] + [token for token in tokens[1:-1] if len(token) > 1] + [tokens[-1]]

    return " ".join(tokens).strip()


def normalize_generic(name: str) -> str:
    """Canonical form for entities with no special naming conventions."""
    working = strip_accents(name).lower().strip()
    working = _AMPERSAND.sub(" and ", working)
    working = _PUNCTUATION.sub(" ", working)
    return _WHITESPACE.sub(" ", working).strip()


def name_tokens(canonical: str) -> frozenset[str]:
    """Token set of a canonical name, used for blocking and overlap scoring."""
    return frozenset(token for token in canonical.split() if token)


def blocking_key(canonical: str) -> str:
    """Cheap key that groups plausible matches together.

    Comparing every candidate against every existing entity is quadratic and
    unusable at graph scale. Blocking on the first token narrows comparison to
    a handful, at the cost of missing pairs that disagree on their first word —
    an acceptable trade, since identifier matching catches the important ones.
    """
    tokens = canonical.split()
    return tokens[0] if tokens else ""


__all__ = [
    "blocking_key",
    "name_tokens",
    "normalize_company_name",
    "normalize_generic",
    "normalize_person_name",
    "strip_accents",
]
