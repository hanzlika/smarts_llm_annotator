import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from smarts_llm_annotator.prompting import (
    build_prompt,
    get_top_n_most_similar_smarts_description_examples,
    get_top_n_most_similar_smarts_description_examples_wrapper,
)

MORGAN_GEN = GetMorganGenerator(radius=2, fpSize=1024)


def _fp(smarts):
    mol = Chem.MolFromSmarts(smarts)
    Chem.SanitizeMol(mol)
    return MORGAN_GEN.GetFingerprint(mol)


def _ref_df():
    return pd.DataFrame(
        {
            "smarts": ["CC", "CCC", "c1ccccc1"],
            "cleaned_description": ["ethane", "propane", "benzene"],
            "fps": [_fp("CC"), _fp("CCC"), _fp("c1ccccc1")],
        }
    )


def test_most_similar_ranks_identical_pattern_first():
    ref_df = _ref_df()
    examples = get_top_n_most_similar_smarts_description_examples(_fp("CC"), ref_df, n=2)
    assert examples[0] == ("CC", "ethane")
    assert len(examples) == 2


def test_missing_fingerprint_returns_no_examples():
    # regression test: this used to fall back to a *random* sample of the
    # reference set, presented in the prompt as if genuinely similar --
    # actively misleading. An invalid query SMARTS should get no examples.
    ref_df = _ref_df()
    examples = get_top_n_most_similar_smarts_description_examples(None, ref_df, n=1)
    assert examples == []


def test_wrapper_reads_reference_csv_and_returns_one_list_per_row(tmp_path):
    ref_path = tmp_path / "smarts_examples.csv"
    ref_path.write_text("smarts,cleaned_description\nCC,ethane\nCCC,propane\n")

    df = pd.DataFrame({"smarts": ["CC", "c1ccccc1"]})
    results = get_top_n_most_similar_smarts_description_examples_wrapper(
        df, "smarts", ref_path=ref_path, n=1
    )

    assert len(results) == len(df)
    assert results[0] == [("CC", "ethane")]


def test_build_prompt_with_literal_lists():
    prompt = build_prompt(
        ["methane", "ethane"],
        [("CC", "ethane"), ("C", "methane")],
        "CCC",
    )
    assert "methane" in prompt
    assert "Input: CC -> Output: ethane" in prompt
    assert prompt.strip().endswith("Input: CCC -> Output:")


def test_build_prompt_with_stringified_lists():
    prompt = build_prompt(
        "['methane', 'ethane']",
        "[('CC', 'ethane')]",
        "CCC",
    )
    assert "methane" in prompt
    assert "Input: CC -> Output: ethane" in prompt


def test_build_prompt_with_unparseable_input_falls_back_to_empty():
    prompt = build_prompt("not a list", "also not a list", "CCC")
    assert "Input: CCC -> Output:" in prompt


def test_build_prompt_omits_examples_header_when_no_examples():
    prompt = build_prompt([], [], "CCC")
    assert "similar SMARTS-to-name conversions" not in prompt
