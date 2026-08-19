"""
smarts_annotation_pipeline.py

End-to-end pipeline:
1. SMARTS
2. PubChem CID lookup
3. smallest-MW IUPAC names
4. similar labeled examples
5. LLM-generated human-readable name

Assumes the input CSV has already been preprocessed into a column of
valid SMARTS strings (see e.g. chemspace_utils.chemspace_label_to_smarts
for one such preprocessing step).
"""

import argparse
import asyncio
from pathlib import Path

import pandas as pd

from scripts import llm_utils, name_formatting, prompting, pubchem_lookup

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUT_PATH = DATA_DIR / "output.csv"


async def run(
    in_path,
    smarts_col,
    out_path=DEFAULT_OUT_PATH,
    model_name=None,
    ignore_time_window=False,
):
    """
    Run the full annotation pipeline over the SMARTS in in_path[smarts_col]
    and write the results (including LLM-generated names) to out_path.

    PubChem lookups respect NCBI's requested off-peak window (weekday
    9pm-5am ET, or anytime on weekends) unless ignore_time_window=True.

    If model_name isn't given, it's resolved from the API's available
    models: prompted for interactively in a terminal, otherwise defaulted
    to the first model listed (see llm_utils.resolve_model_name).
    """
    model_name = await llm_utils.resolve_model_name(model_name)

    df = pd.read_csv(in_path)[[smarts_col]].dropna()

    # PubChem CID lookup + smallest-MW IUPAC names, via the live API
    df = pubchem_lookup.lookup_smallest_mw_from_smarts(
        df,
        smarts_col=smarts_col,
        limit_per_smarts=1000,
        max_workers=8,
        show_progress=True,
        ignore_time_window=ignore_time_window,
    )

    # most similar labeled SMARTS examples, for few-shot prompting
    df["similar_examples"] = prompting.get_top_n_most_similar_smarts_description_examples_wrapper(
        df, smarts_col, n=10
    )

    # compile prompts
    df["prompt"] = df.apply(
        lambda row: prompting.build_prompt(
            row["best_IUPAC_names"], row["similar_examples"], row[smarts_col]
        ),
        axis=1,
    )

    # LLM call
    out_df = await llm_utils.run_llm_on_dataframe(df, model_name=model_name, prompt_column="prompt")

    # normalize the generated name to this project's display format (lowercase,
    # ASCII, no catalog-style disambiguation suffixes) -- see name_formatting.py.
    # Left untouched on error rows, where llm_output is an "Error: ..." message.
    is_ok = out_df["llm_status"] == "ok"
    out_df.loc[is_ok, "llm_output"] = out_df.loc[is_ok, "llm_output"].apply(name_formatting.normalize_name)

    out_df[[smarts_col, "best_IUPAC_names", "llm_output"]].to_csv(out_path, index=False)
    return out_df


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate SMARTS patterns with LLM-generated human-readable names."
    )
    parser.add_argument("in_path", help="Input CSV containing a column of SMARTS patterns.")
    parser.add_argument("smarts_col", help="Name of the column containing SMARTS patterns.")
    parser.add_argument("--out-path", default=str(DEFAULT_OUT_PATH), help="Output CSV path.")
    parser.add_argument(
        "--model",
        default=None,
        help="Model to use for naming. If omitted, prompts you to choose from "
        "the models available through your API key (or picks the first one "
        "when not running in an interactive terminal).",
    )
    parser.add_argument(
        "--ignore-time-window",
        action="store_true",
        help="Skip NCBI's requested off-peak (9pm-5am ET / weekend) window for PubChem calls.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(run(args.in_path, args.smarts_col, args.out_path, args.model, args.ignore_time_window))
