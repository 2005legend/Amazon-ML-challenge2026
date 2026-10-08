import polars as pl

from ber.genrules import boost, decoy_word_pairs, is_abbrev, noise_addition_pairs, numgate_pairs, rel, top1


def test_rel_classifies_street_number_edits():
    assert rel("123", "123") == "eq"
    assert rel("123", "1234") == "indel"
    assert rel("123", "124") == "sub"
    assert rel("123", "132") == "transpose"
    assert rel("120", "117") == "near<=4"
    assert rel("120", "140") == "sub"  # one differing digit is a substitution before any distance check
    assert rel("120", "135") == "near<=20"
    assert rel("120", "480") == "far"
    assert rel("", "12") == "na"


def test_is_abbrev_needs_same_first_letter_and_subsequence():
    assert is_abbrev("cie", "compagnie")
    assert not is_abbrev("compagnie", "cie")
    assert not is_abbrev("ice", "compagnie")
    assert not is_abbrev("compag", "compagnie")  # longer than 5 characters


def test_boost_raises_selected_scores_only():
    p = pl.DataFrame({"s1_id": ["a", "a"], "tgt_id": ["x", "y"], "p2": [0.2, 0.99]})
    out = boost(p, pl.DataFrame({"s1_id": ["a", "a"], "tgt_id": ["x", "y"]}))
    assert out.sort("tgt_id")["p2"].to_list() == [0.95, 0.99]


def test_top1_keeps_the_targets_best_claimant():
    sc = pl.DataFrame({"s1_id": ["a", "b"], "tgt_id": ["x", "x"], "p2": [0.3, 0.8]})
    pairs = pl.DataFrame({"s1_id": ["a", "b"], "tgt_id": ["x", "x"]})
    assert top1(pairs, sc)["s1_id"].to_list() == ["b"]


def _edits(words, same, n):
    return pl.DataFrame({"s1_id": [f"s{i}" for i in range(n)], "tgt_id": [f"t{i}" for i in range(n)], "country": ["US"] * n,
                         "a_tset": [1.0 if same else 0.85] * n, "nf_eq": [1.0 if same else 0.0] * n,
                         "add": [[words]] * n, "drop": [[]] * n})


def test_decoy_word_needs_neighbour_dominance_and_different_number():
    x = pl.concat([_edits("holdings", False, 1200), _edits("holdings", True, 2)])
    flagged = decoy_word_pairs(x)
    assert flagged.height == 1200  # the same-address pairs are never flagged


def test_noise_addition_needs_same_address_dominance():
    x = pl.concat([_edits("services", True, 1500), _edits("services", False, 100)])
    assert noise_addition_pairs(x).height == 1500


def test_numgate_only_for_gated_countries_with_disjoint_numbers():
    f = pl.DataFrame({"s1_id": ["a", "b", "c"], "tgt_id": ["x", "y", "z"], "country": ["France", "France", "US"],
                      "nf_eq": [0.0, 0.0, 0.0], "num_overlap": [0.0, 1.0, 0.0], "num_jacc": [0.0, 0.5, 0.0]})
    assert numgate_pairs(f, ["France"])["s1_id"].to_list() == ["a"]
