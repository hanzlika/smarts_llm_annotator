from unittest.mock import AsyncMock, patch

import pandas as pd

from smarts_llm_annotator import smarts_annotation_pipeline


def _fake_lookup(df, smarts_col, **kwargs):
    out = df.copy()
    out["cids"] = [[1, 2] for _ in range(len(out))]
    out["best_IUPAC_names"] = [["methanol", "ethanol"] for _ in range(len(out))]
    return out


async def _fake_llm(df, model_name=None, prompt_column=None):
    out = df.copy()
    out["llm_output"] = "some name (6)"  # exercises the normalize_name step too
    out["llm_status"] = "ok"
    return out


def _patches():
    return (
        patch(
            "smarts_llm_annotator.pubchem_lookup.lookup_smallest_mw_from_smarts",
            side_effect=_fake_lookup,
        ),
        patch("smarts_llm_annotator.llm_utils.run_llm_on_dataframe", side_effect=_fake_llm),
        patch(
            "smarts_llm_annotator.llm_utils.resolve_model_name",
            new=AsyncMock(return_value="fake-model"),
        ),
    )


async def test_run_accepts_a_dataframe_directly(tmp_path):
    df = pd.DataFrame({"smarts": ["CC", "CCC"]})
    out_path = tmp_path / "out.csv"

    p1, p2, p3 = _patches()
    with p1, p2, p3:
        result = await smarts_annotation_pipeline.run(df, smarts_col="smarts", out_path=out_path)

    assert list(result["smarts"]) == ["CC", "CCC"]
    # normalize_name should have stripped the catalog-index "(6)"
    assert (result["llm_output"] == "some name").all()
    assert out_path.exists()
    written = pd.read_csv(out_path)
    assert list(written.columns) == ["smarts", "best_IUPAC_names", "llm_output"]
    assert (written["llm_output"] == "some name").all()


async def test_run_accepts_a_csv_path(tmp_path):
    in_path = tmp_path / "in.csv"
    pd.DataFrame({"smarts": ["CC", None, "CCC"]}).to_csv(in_path, index=False)
    out_path = tmp_path / "out.csv"

    p1, p2, p3 = _patches()
    with p1, p2, p3:
        result = await smarts_annotation_pipeline.run(str(in_path), smarts_col="smarts", out_path=out_path)

    # the row with a missing smarts value should have been dropped
    assert list(result["smarts"]) == ["CC", "CCC"]
