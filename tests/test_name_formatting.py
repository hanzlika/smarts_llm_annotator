from smarts_llm_annotator.name_formatting import fix_unbalanced_leading_paren, normalize_name


def test_lowercases():
    assert normalize_name("Aromatic Nitrogen") == "aromatic nitrogen"


def test_spells_out_greek_letters():
    assert normalize_name("α,β-unsaturated aldehyde") == "alpha,beta-unsaturated aldehyde"


def test_strips_trailing_numeric_disambiguation_index():
    assert normalize_name("coumarins (6)") == "coumarins"


def test_spells_out_trailing_comparison_thresholds_instead_of_discarding_them():
    assert normalize_name("too many cooh groups (>1)") == "too many cooh groups with more than one"
    assert normalize_name("high halogen content (>3)") == "high halogen content with more than three"
    assert normalize_name("some filter (<2)") == "some filter with less than two"
    assert normalize_name("some filter (>=2)") == "some filter with at least two"
    assert normalize_name("some filter (<=2)") == "some filter with at most two"


def test_comparison_threshold_falls_back_to_digit_outside_spelled_range():
    assert normalize_name("some filter (>15)") == "some filter with more than 15"


def test_strips_trailing_qualifier_words():
    assert normalize_name("quinone (additional)") == "quinone"
    assert normalize_name("some filter (specific)") == "some filter"
    assert normalize_name("some filter (GENERAL)") == "some filter"


def test_strips_multiple_stacked_trailing_groups():
    assert normalize_name("activated alkene (michael acceptor) (2)") == "activated alkene (michael acceptor)"


def test_preserves_meaningful_parentheticals():
    assert normalize_name("sulphate esters (thio)") == "sulphate esters (thio)"
    assert normalize_name("activated alkene (michael acceptor)") == "activated alkene (michael acceptor)"


def test_preserves_genuine_chemical_content_that_looks_like_a_qualifier():
    # "(c<5)" is a real carbon-count constraint, not a disambiguation index
    text = "alkyl (c<5) or benzylester of sulphonic or phosphonic acid"
    assert normalize_name(text) == text


def test_preserves_genuine_locants():
    assert normalize_name("2,3-dihydro-1h-phenalene derivatives") == "2,3-dihydro-1h-phenalene derivatives"


def test_collapses_whitespace():
    assert normalize_name("  too   many   spaces  ") == "too many spaces"


def test_non_string_input_passed_through():
    assert normalize_name(None) is None


def test_fix_unbalanced_leading_paren():
    assert fix_unbalanced_leading_paren("(poly(azo(anthracene))") == "poly(azo(anthracene))"


def test_fix_unbalanced_leading_paren_is_noop_when_balanced():
    assert fix_unbalanced_leading_paren("sulphate esters (thio)") == "sulphate esters (thio)"
