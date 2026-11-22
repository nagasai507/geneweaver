from src.genomics.fasta_parser import parse_fasta
from src.alignment.cpu_baseline import find_off_targets
from src.alignment.dask_accelerated import find_off_targets_dask


def test_fasta_parser(tmp_path):
    fasta_file = tmp_path / "test.fasta"

    fasta_file.write_text(
        ">Gene_A\nATGCGTACGTAG\n"
        ">Gene_B\nGCTAGCTAGCAT\n"
    )

    sequences = parse_fasta(str(fasta_file))

    assert len(sequences) == 2
    assert sequences[0]["id"] == "Gene_A"
    assert sequences[0]["length"] == 12


def test_exact_match():
    sequences = [{
        "id": "Gene_A",
        "sequence": "ATGCGTACGTAG",
    }]

    results = find_off_targets(sequences, "ATGCGTACGTAG", 0)

    assert len(results) == 1
    assert results[0]["mismatches"] == 0
    assert results[0]["risk"] == "HIGH"


def test_mismatch_detection():
    sequences = [{
        "id": "Gene_B",
        "sequence": "ATGCGTACGTAA",
    }]

    results = find_off_targets(sequences, "ATGCGTACGTAG", 1)

    assert len(results) == 1
    assert results[0]["mismatches"] == 1
    assert results[0]["risk"] == "MEDIUM"


def test_no_match():
    sequences = [{
        "id": "Gene_C",
        "sequence": "CCCCCCCCCCCC",
    }]

    results = find_off_targets(sequences, "ATGCGTACGTAG", 0)

    assert len(results) == 0


def test_empty_sequence():
    sequences = [{
        "id": "Gene_D",
        "sequence": "",
    }]

    results = find_off_targets(sequences, "ATGCGTACGTAG", 2)

    assert len(results) == 0


def test_numba_matches_cpu_results():
    from src.alignment.numba_accelerated import find_off_targets_numba

    sequences = [{
        "id": "Gene_A",
        "sequence": "ATGCGTACGTAG",
    }]

    cpu_results = find_off_targets(
        sequences, "ATGCGTACGTAG", 2
    )

    numba_results = find_off_targets_numba(
        sequences, "ATGCGTACGTAG", 2
    )

    assert numba_results == cpu_results
    

def test_empty_guide_sequence():
    sequences = [{
        "id": "Gene_A",
        "sequence": "ATGCGTACGTAG",
    }]

    import pytest

    with pytest.raises(ValueError):
        find_off_targets(sequences, "", 2)


def test_negative_mismatch_limit():
    sequences = [{
        "id": "Gene_A",
        "sequence": "ATGCGTACGTAG",
    }]

    import pytest

    with pytest.raises(ValueError):
        find_off_targets(sequences, "ATGCGTACGTAG", -1)


def test_dask_matches_cpu_results():
    sequences = [
        {"id": "Gene_A", "sequence": "ATGCGTACGTAG"},
        {"id": "Gene_B", "sequence": "ATGCGTACGTAA"},
    ]

    cpu_results = find_off_targets(
        sequences, "ATGCGTACGTAG", 2
    )

    dask_results = find_off_targets_dask(
        sequences, "ATGCGTACGTAG", 2
    )

    assert dask_results == cpu_results


def test_dask_empty_guide_sequence():
    import pytest

    sequences = [{"id": "Gene_A", "sequence": "ATGCGTACGTAG"}]

    with pytest.raises(ValueError):
        find_off_targets_dask(sequences, "", 2)


def test_dask_negative_mismatch_limit():
    import pytest

    sequences = [{"id": "Gene_A", "sequence": "ATGCGTACGTAG"}]

    with pytest.raises(ValueError):
        find_off_targets_dask(sequences, "ATGCGTACGTAG", -1)


def test_dask_empty_gene_list():
    results = find_off_targets_dask(
        [], "ATGCGTACGTAG", 2
    )

    assert results == []


def test_dask_multiple_genes_with_no_match():
    sequences = [
        {"id": "Gene_A", "sequence": "ATGCGTACGTAG"},
        {"id": "Gene_B", "sequence": "CCCCCCCCCCCC"},
        {"id": "Gene_C", "sequence": "ATGCGTACGTAA"},
    ]

    cpu_results = find_off_targets(
        sequences, "ATGCGTACGTAG", 2
    )

    dask_results = find_off_targets_dask(
        sequences, "ATGCGTACGTAG", 2
    )

    assert dask_results == cpu_results


def test_fasta_parser_preserves_unknown_base(tmp_path):
    fasta_file = tmp_path / "unknown.fasta"
    fasta_file.write_text(">Gene_A\nATGCNATGC\n")

    sequences = parse_fasta(str(fasta_file))

    assert sequences[0]["sequence"] == "ATGCNATGC"
    assert sequences[0]["length"] == 9


def test_fasta_parser_rejects_invalid_base(tmp_path):
    import pytest

    fasta_file = tmp_path / "invalid.fasta"
    fasta_file.write_text(">Gene_A\nATGCXATGC\n")

    with pytest.raises(ValueError, match="invalid DNA base"):
        parse_fasta(str(fasta_file))


def test_fasta_parser_accepts_valid_dna(tmp_path):
    fasta_file = tmp_path / "valid.fasta"
    fasta_file.write_text(">Gene_A\nATGCATGC\n")

    sequences = parse_fasta(str(fasta_file))

    assert sequences[0]["sequence"] == "ATGCATGC"
    assert sequences[0]["length"] == 8
