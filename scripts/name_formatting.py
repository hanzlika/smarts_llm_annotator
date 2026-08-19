"""
name_formatting.py

Canonical display formatting for SMARTS names. Used both to clean the
few-shot reference examples (scripts/prompting.py) and to sanitize
LLM-generated names before they're written out / displayed. Sharing one
implementation means the examples the LLM is shown already comply with
the same rules enforced on its own output, instead of competing signals.
"""

import re

# Spelled out so names stay plain ASCII (safer for search/sort/copy-paste
# in a web UI than relying on every font/consumer handling Greek glyphs).
_GREEK_TO_LATIN = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ω": "omega",
}

# Trailing catalog-index markers, e.g. "coumarins (6)", "quinone (additional)".
# These aren't part of the name -- they distinguish catalog variants of
# the same base filter (a plain number carries no meaning of its own).
# Genuine chemical parentheticals like "(thio)" or "(michael acceptor)"
# are left untouched.
_TRAILING_INDEX_RE = re.compile(r"\s*\(\s*\d+\s*\)\s*$")
_TRAILING_QUALIFIER_RE = re.compile(r"\s*\(\s*(?:specific|additional|general)\s*\)\s*$", re.IGNORECASE)

# A trailing *comparison* threshold, e.g. "too many cooh groups (>1)", is
# real quantitative content, unlike a bare catalog index -- unlike a
# genuinely supplementary aside (e.g. "(thio)"), the count IS the rule, so
# fold it into the phrase as plain words rather than leaving it parenthetical.
# Some entries in the source data already do this by hand (e.g. "more than
# one"), so numbers are spelled out too, matching that existing convention.
_TRAILING_COMPARISON_RE = re.compile(r"\(\s*(>=|<=|>|<)\s*(\d+)\s*\)\s*$")
_COMPARATOR_WORDS = {
    ">=": "at least",
    "<=": "at most",
    ">": "more than",
    "<": "less than",
}
_NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
}


def _spell_out_comparison(match: re.Match) -> str:
    number = match.group(2)
    number_word = _NUMBER_WORDS.get(number, number)
    return f"with {_COMPARATOR_WORDS[match.group(1)]} {number_word}"


def normalize_name(name: str) -> str:
    """
    Normalize a SMARTS name to this project's display format: lowercase,
    ASCII (Greek letters spelled out), no trailing catalog-index
    parentheticals (a trailing comparison threshold like "(>1)" is folded
    into the phrase as plain words instead, e.g. "with more than one", so
    that information isn't lost), collapsed whitespace.
    """
    if not isinstance(name, str):
        return name

    text = name.strip().lower()
    for greek, latin in _GREEK_TO_LATIN.items():
        text = text.replace(greek, latin)

    # spell out a trailing comparison threshold (e.g. "(>1)" -> "(more than 1)")
    # before stripping catalog-index noise, so real quantitative info survives.
    text = _TRAILING_COMPARISON_RE.sub(_spell_out_comparison, text)

    # A description can carry more than one trailing group, e.g.
    # "activated alkene (michael acceptor) (2)".
    for _ in range(3):
        stripped = _TRAILING_INDEX_RE.sub("", text)
        stripped = _TRAILING_QUALIFIER_RE.sub("", stripped)
        if stripped == text:
            break
        text = stripped

    return re.sub(r"\s+", " ", text).strip()


def fix_unbalanced_leading_paren(name: str) -> str:
    """
    Fix descriptions missing an outer closing parenthesis but carrying one
    stray leading '(' instead, e.g. "(poly(azo(anthracene))" -> drop the
    leading '(' so the groups balance.
    """
    if isinstance(name, str) and name.startswith("(") and name.count("(") == name.count(")") + 1:
        return name[1:].strip()
    return name
