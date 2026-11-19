
from benchmarks.benchmark import create_test_data


def test_benchmark_creates_three_genes():
    sequences, guide = create_test_data()

    assert len(sequences) == 3


def test_benchmark_gene_lengths_and_known_matches():
    sequences, guide = create_test_data()

    for gene in sequences:
        dna = gene["sequence"]

        assert len(dna) == 50000
        assert dna[1000:1012] == guide
        assert dna[25000:25012] == guide


def test_benchmark_data_is_reproducible():
    first_sequences, first_guide = create_test_data()
    second_sequences, second_guide = create_test_data()

    assert first_guide == second_guide
    assert first_sequences == second_sequences
