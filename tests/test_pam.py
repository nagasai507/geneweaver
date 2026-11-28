import pytest

from src.biological.pam import reverse_complement, find_pam_sites
from src.biological.pam import is_valid_pam


def test_valid_ngg_pam():
    assert is_valid_pam("AGG") is True


def test_invalid_ngg_pam():
    assert is_valid_pam("AGA") is False


def test_custom_pam_pattern():
    assert is_valid_pam("TTT", "TTT") is True


def test_lowercase_pam():
    assert is_valid_pam("agg") is True


def test_invalid_dna_base():
    assert is_valid_pam("AXG") is False


def test_pam_length_mismatch():
    assert is_valid_pam("AG", "NGG") is False


def test_empty_pam_pattern():
    with pytest.raises(ValueError):
        is_valid_pam("AGG", "")


def test_invalid_pattern_code():
    with pytest.raises(ValueError):
        is_valid_pam("AGG", "XYZ")


def test_non_string_input():
    with pytest.raises(ValueError):
        is_valid_pam(123)


def test_reverse_complement():
    assert reverse_complement("ATGC") == "GCAT"


def test_forward_strand_detection():
    sequence = "A" * 20 + "AGG"

    results = find_pam_sites(sequence)

    assert len(results) == 1
    assert results[0]["strand"] == "+"
    assert results[0]["pam"] == "AGG"
    assert results[0]["start"] == 0
    assert results[0]["end"] == 23


def test_reverse_strand_detection():
    sequence = "CCA" + "A" * 20

    results = find_pam_sites(sequence)

    assert len(results) == 1
    assert results[0]["strand"] == "-"
    assert results[0]["pam"] == "TGG"
    assert results[0]["start"] == 0
    assert results[0]["end"] == 23


def test_no_pam_found():
    sequence = "A" * 30

    results = find_pam_sites(sequence)

    assert results == []


def test_reverse_complement_invalid_base():
    with pytest.raises(ValueError):
        reverse_complement("ATGX")


def test_find_pam_invalid_dna():
    with pytest.raises(ValueError):
        find_pam_sites("ATGCXYZ")


def test_zero_protospacer_length():
    with pytest.raises(ValueError):
        find_pam_sites("ATGC", protospacer_length=0)


def test_invalid_protospacer_length():
    with pytest.raises(ValueError):
        find_pam_sites("ATGC", protospacer_length="20")


def test_custom_pam_scanning():
    sequence = "C" * 20 + "TTT"

    results = find_pam_sites(sequence, pam_pattern="TTT")

    assert len(results) == 1
    assert results[0]["strand"] == "+"
    assert results[0]["pam"] == "TTT"