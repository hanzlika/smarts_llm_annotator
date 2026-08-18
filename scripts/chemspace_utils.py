# -*- coding: utf-8 -*-
"""
chemspace_utils.py

Helpers for preprocessing Chemspace-formatted substructure labels into
valid SMARTS strings.
"""

import re

import pandas as pd
from rdkit import Chem


def chemspace_label_to_smarts(label: str) -> str:
    """
    Convert a Chemspace substructure label into a SMARTS-compatible string.

    Chemspace uses '[R]' synonymously with a wildcard atom, and marks
    aromaticity/aliphaticity with an 'ar'/'al' suffix right after the atom
    letter (e.g. 'Car', 'Nal') instead of SMARTS' lower/upper case convention.
    """
    # Chemspace uses [R]'s pretty much synonymously to wildcard atoms
    label = re.sub(r'\[R\]', '*', label)

    def repl(match):
        # (.) captures the previous character, (ar|al) captures the suffix.
        # Lowercasing/uppercasing that character makes it SMARTS-friendly
        # without losing information.
        prev_char = match.group(1)
        pattern = match.group(2)
        if pattern == 'ar':
            return prev_char.lower()
        elif pattern == 'al':
            return prev_char.upper()

    return re.sub(r'(.)((?:ar|al))', repl, label)


def chemspace_labels_to_smarts(labels: pd.Series) -> pd.Series:
    """
    Convert a column of Chemspace labels into valid, RDKit-canonicalized
    SMARTS strings (regex conversion, then a parse/re-serialize pass so
    downstream code always sees valid SMARTS).
    """
    return labels.apply(chemspace_label_to_smarts).apply(
        lambda x: Chem.MolToSmarts(Chem.MolFromSmarts(x, mergeHs=True))
    )
