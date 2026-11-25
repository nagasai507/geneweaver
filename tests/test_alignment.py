import pytest

from src.genomics.fasta_parser import parse_fasta
from src.alignment.cpu_baseline import find_off_targets
from src.alignment.dask_accelerated import find_off_targets_dask
from src.alignment.numba_accelerated import find_off_targets_numba


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


@pytest.mark.parametrize(
    "finder",
    [
        find_off_targets,
        find_off_targets_numba,
        find_off_targets_dask,
    ],
)
@pytest.mark.parametrize("guide", ["ATGCXATG", "ATGCNATG"])


def test_alignment_rejects_invalid_guide_bases(finder, guide):
    sequences = [{
        "id": "Gene_A",
        "sequence": "ATGCATGC",
    }]

    with pytest.raises(ValueError, match="invalid DNA base"):
        finder(sequences, guide, 2)

@pytest.mark.parametrize(
    "finder",
    [
        find_off_targets,
        find_off_targets_numba,
        find_off_targets_dask,
    ],
)


def test_alignment_accepts_lowercase_guide(finder):
    sequences = [{
        "id": "Gene_A",
        "sequence": "ATGCATGC",
    }]

    results = finder(sequences, "atgcatgc", 0)

    assert len(results) == 1
    assert results[0]["mismatches"] == 0

def test_fasta_parser_rejects_empty_gene_sequence(tmp_path):
    fasta_file = tmp_path / "empty_gene.fasta"
    fasta_file.write_text(">Gene_A\n")

    with pytest.raises(ValueError, match="empty DNA sequence"):
        parse_fasta(str(fasta_file))

@pytest.mark.parametrize(
    "finder",
    [
        find_off_targets,
        find_off_targets_numba,
        find_off_targets_dask,
    ],
)
@pytest.mark.parametrize(
    "max_mismatches",
    [-1, 2.5, "two", True],
)
def test_alignment_rejects_invalid_mismatch_limits(
    finder,
    max_mismatches,
):
    sequences = [
        {"id": "Gene_A", "sequence": "ATGCATGC"}
    ]

    with pytest.raises(
        ValueError,
        match="Maximum mismatches must be a non-negative integer",
    ):
        finder(sequences, "ATGC", max_mismatches)

@pytest.mark.parametrize(
    "finder",
    [
        find_off_targets,
        find_off_targets_numba,
        find_off_targets_dask,
    ],
)
@pytest.mark.parametrize(
    "guide",
    [12345, None, ["A", "T", "G", "C"]],
)
def test_alignment_rejects_non_string_guide(finder, guide):
    sequences = [
        {"id": "Gene_A", "sequence": "ATGCATGC"}
    ]

    with pytest.raises(
        ValueError,
        match="Guide sequence must be a string",
    ):
        finder(sequences, guide, 2)