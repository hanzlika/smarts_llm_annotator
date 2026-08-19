"""
Regression tests guarding the cleanup done by smarts_llm_annotator/clean_reference_examples.py
-- catches new raw data being merged into data/smarts_examples.csv without
being run through the cleaner.
"""

import pandas as pd

from smarts_llm_annotator.name_formatting import normalize_name
from smarts_llm_annotator.prompting import DEFAULT_SMARTS_EXAMPLES_PATH


def _reference_df():
    return pd.read_csv(DEFAULT_SMARTS_EXAMPLES_PATH)


def test_no_description_is_just_the_raw_smarts():
    df = _reference_df()
    corrupted = df[df["cleaned_description"].str.lower() == df["smarts"].str.lower()]
    assert corrupted.empty, f"description == smarts for: {corrupted['smarts'].tolist()}"


def test_descriptions_are_already_normalized():
    df = _reference_df()
    not_normalized = df[df["cleaned_description"] != df["cleaned_description"].apply(normalize_name)]
    assert not_normalized.empty, f"not normalized: {not_normalized['cleaned_description'].tolist()}"


def test_no_exact_duplicate_rows():
    df = _reference_df()
    dupes = df[df.duplicated(subset=["smarts", "cleaned_description"], keep=False)]
    assert dupes.empty, f"duplicate (smarts, description) rows: {len(dupes)}"
