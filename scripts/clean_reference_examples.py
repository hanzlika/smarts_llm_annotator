# -*- coding: utf-8 -*-
"""
clean_reference_examples.py

One-off maintenance script that cleans data/smarts_examples.csv (the
few-shot reference set used by scripts/prompting.py) so its descriptions
already comply with the display format enforced on LLM output -- see
scripts/name_formatting.py. Re-run this after adding new raw reference
rows.

Fixes applied:
  - normalize_name(): lowercase, spell out Greek letters, drop trailing
    catalog disambiguation markers like "(6)"/"(additional)".
  - fix_unbalanced_leading_paren(): repair descriptions missing their
    outer closing parenthesis.
  - drop rows where the description is actually just the raw SMARTS
    pattern (a data-entry bug, not a name).
  - drop exact-duplicate (smarts, cleaned description) rows.
"""

from pathlib import Path

import pandas as pd

from scripts.name_formatting import fix_unbalanced_leading_paren, normalize_name

REFERENCE_PATH = Path(__file__).resolve().parents[1] / "data" / "smarts_examples.csv"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[df["cleaned_description"].str.lower().str.strip() != df["smarts"].str.lower().str.strip()]

    df["cleaned_description"] = df["cleaned_description"].apply(fix_unbalanced_leading_paren)
    df["cleaned_description"] = df["cleaned_description"].apply(normalize_name)

    df = df.drop_duplicates(subset=["smarts", "cleaned_description"])

    return df.reset_index(drop=True)


if __name__ == "__main__":
    before = pd.read_csv(REFERENCE_PATH)
    after = clean(before)
    after.to_csv(REFERENCE_PATH, index=False)
    print(f"{len(before)} -> {len(after)} rows ({len(before) - len(after)} removed)")
