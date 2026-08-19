import pandas as pd

from scripts.chemspace_utils import chemspace_label_to_smarts, chemspace_labels_to_smarts


def test_wildcard_replacement():
    assert chemspace_label_to_smarts("[R]C(=O)N") == "*C(=O)N"


def test_aromatic_suffix_lowercases_atom():
    assert chemspace_label_to_smarts("Car") == "c"


def test_aliphatic_suffix_uppercases_atom():
    assert chemspace_label_to_smarts("cal") == "C"


def test_aliphatic_suffix_is_noop_on_already_uppercase_atom():
    assert chemspace_label_to_smarts("Nal") == "N"


def test_wildcard_and_suffix_combined():
    assert chemspace_label_to_smarts("[R]Car") == "*c"


def test_repeated_aromatic_ring():
    assert chemspace_label_to_smarts("CarCarCarCarCarCar1") == "cccccc1"


def test_labels_to_smarts_produces_valid_canonical_smarts():
    labels = pd.Series(["[Nar]", "O=[C]([R])[N]([R])[R]"])
    result = chemspace_labels_to_smarts(labels)
    assert result.tolist() == ["n", "O=C(*)N(*)*"]
