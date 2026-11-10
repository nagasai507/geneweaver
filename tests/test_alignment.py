from src.genomics.fasta_parser import parse_fasta
from src.alignment.cpu_baseline import find_off_targets


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
