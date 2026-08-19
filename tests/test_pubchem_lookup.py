import pandas as pd

from smarts_llm_annotator.pubchem_lookup import get_lowest_mw_iupac_names


def _compounds_properties():
    return pd.DataFrame(
        {
            "CID": [1, 2, 3, 4, 5],
            "MolecularWeight": [100, 50, 75, 30, 20],
            "IUPACName": ["a", "b", "c", "d", "e"],
        }
    )


def test_sorts_by_ascending_molecular_weight():
    df = pd.DataFrame({"cids": [[1, 2, 3]]})
    result = get_lowest_mw_iupac_names(df, _compounds_properties())
    assert result == [["b", "c", "a"]]


def test_parses_stringified_cid_lists():
    df = pd.DataFrame({"cids": ["[4, 5]"]})
    result = get_lowest_mw_iupac_names(df, _compounds_properties())
    assert result == [["e", "d"]]


def test_empty_cids_yield_empty_list():
    df = pd.DataFrame({"cids": [[], "not a list"]})
    result = get_lowest_mw_iupac_names(df, _compounds_properties())
    assert result == [[], []]


def test_truncates_to_n():
    df = pd.DataFrame({"cids": [[1, 2, 3]]})
    result = get_lowest_mw_iupac_names(df, _compounds_properties(), n=2)
    assert result == [["b", "c"]]


def test_returns_a_list_per_input_row_not_a_dataframe_column():
    # regression test: get_lowest_mw_iupac_names used to return df['best_IUPAC_names'],
    # a column that doesn't exist at the call site, raising KeyError.
    df = pd.DataFrame({"cids": [[1], [2]]})
    result = get_lowest_mw_iupac_names(df, _compounds_properties())
    assert isinstance(result, list)
    assert len(result) == 2
