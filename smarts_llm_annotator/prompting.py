"""
prompting.py

Builds the LLM prompt used to name a SMARTS pattern: finds the most
similar SMARTS patterns in a labeled reference set (by Morgan fingerprint
similarity) and combines them with candidate IUPAC names into a few-shot
prompt.
"""

import ast
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from tqdm import tqdm

DEFAULT_SMARTS_EXAMPLES_PATH = Path(__file__).resolve().parents[1] / "data" / "smarts_examples.csv"

BASE_INSTRUCTIONS = """You are an LLM that converts SMARTS substructure filters into concise, human-readable names.
- **Input**: A SMARTS string.
- **Output**: A lowercase name, as short as possible while staying unambiguous -- a single word is fine (e.g. 'quinone'), and most names are 1-3 words; use more only when the chemistry genuinely needs it.
- The name should help molecular biologists/chemists instantly recognize the chemical feature (e.g., 'amide bond').
- Spell out Greek letters as words (e.g. 'alpha,beta-unsaturated', not 'α,β-unsaturated').
- Do not add parenthetical qualifiers, counters, or disambiguation labels (e.g. not 'coumarin (6)' or '(specific)') -- only include a number when it is a genuine chemical locant (e.g. '1,3-dioxolane').
- Return **ONLY the name** - no prefixes, suffixes, explanations, or formatting.
"""

MATCHING_COMPOUNDS_PROMPT_BASE = (
    "IUPAC names of smallest matching compounds (by atom count). Use for inspiration but note:\n"
    "- SMARTS patterns are often simpler than matching compounds\n"
    "- NEVER directly use compound names; identify general features instead:\n"
)

SIMILAR_EXAMPLES_PROMPT_BASE = (
    "\nExamples of similar SMARTS-to-name conversions. Use to maintain consistent naming conventions:\n"
)


def _fingerprints_for(smarts_list, morgan_gen):
    fps = []
    for s in smarts_list:
        mol = Chem.MolFromSmarts(s)
        if mol is None:
            fps.append(None)
            print(f"Invalid SMARTS: {s}")
            continue
        try:
            Chem.SanitizeMol(mol)
            fps.append(morgan_gen.GetFingerprint(mol))
        except ValueError:
            fps.append(None)
            print(f"Fingerprint generation failed for SMARTS: {s}")
    return fps


def get_top_n_most_similar_smarts_description_examples(query_fp, ref_df, n=10):
    """
    Return up to n (smarts, description) pairs from ref_df most similar to
    query_fp. Returns an empty list if query_fp is None (e.g. the query
    SMARTS was invalid) -- unrelated examples presented as "similar" would
    actively mislead the few-shot prompt, so we'd rather show none.
    """
    if query_fp is None:
        return []

    similarities = []
    for _, row in ref_df.iterrows():
        fp2 = row["fps"]
        if fp2 is not None:
            sim = DataStructs.TanimotoSimilarity(query_fp, fp2)
            similarities.append(((row["smarts"], row["cleaned_description"]), sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in similarities[:n]]


def get_top_n_most_similar_smarts_description_examples_wrapper(
    df, smarts_col, ref_path=DEFAULT_SMARTS_EXAMPLES_PATH, n=10
):
    """
    For every SMARTS in df[smarts_col], find the n most similar labeled
    SMARTS examples in the reference file. Returns a list (one entry per
    row of df) of lists of (smarts, description) tuples.
    """
    ref = pd.read_csv(ref_path)
    morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)

    ref["fps"] = _fingerprints_for(ref["smarts"].to_list(), morgan_gen)
    ref = ref.dropna(subset=["fps"])
    smarts_fps = _fingerprints_for(df[smarts_col].to_list(), morgan_gen)

    smarts_examples = []
    for query_fp in tqdm(smarts_fps, desc="Finding most similar smarts examples"):
        smarts_examples.append(
            get_top_n_most_similar_smarts_description_examples(query_fp, ref, n=n)
        )

    return smarts_examples


def build_prompt(best_iupac_names, similar_examples, input_smarts):
    """
    Combine candidate IUPAC names and similar SMARTS-to-name examples into
    a single few-shot prompt for naming input_smarts.

    best_iupac_names / similar_examples may be already-parsed lists, or
    their stringified (repr) form, e.g. as read back from a CSV column.
    """
    def _parse(value, default):
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return default
        return value if value is not None else default

    iupac_list = _parse(best_iupac_names, [])
    examples_list = _parse(similar_examples, [])

    matching_compounds_prompt = (
        MATCHING_COMPOUNDS_PROMPT_BASE + "\n".join(iupac_list[:5]) if iupac_list else ""
    )

    similar_examples_prompt = (
        SIMILAR_EXAMPLES_PROMPT_BASE
        + "".join(f"Input: {smarts} -> Output: {name}\n" for smarts, name in examples_list[:5])
        if examples_list
        else ""
    )

    return (
        BASE_INSTRUCTIONS
        + matching_compounds_prompt
        + similar_examples_prompt
        + f"Now you fill in the final one:\nInput: {input_smarts} -> Output:"
    )
