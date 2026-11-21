from benchmarks.benchmark import create_test_data, run_benchmark


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


def test_benchmark_supports_custom_dataset_size():
    sequences, guide = create_test_data(
        num_genes=10,
        sequence_length=30000
    )

    assert len(sequences) == 10

    for gene in sequences:
        dna = gene["sequence"]

        assert len(dna) == 30000
        assert dna[1000:1012] == guide
        assert dna[25000:25012] == guide


def test_benchmark_rejects_short_sequences():
    import pytest

    with pytest.raises(ValueError):
        create_test_data(sequence_length=10000)


def test_benchmark_rejects_zero_genes():
    import pytest

    with pytest.raises(ValueError):
        create_test_data(num_genes=0)


def test_benchmark_rejects_negative_genes():
    import pytest

    with pytest.raises(ValueError):
        create_test_data(num_genes=-5)


def test_benchmark_rejects_zero_iterations():
    import pytest

    with pytest.raises(ValueError):
        run_benchmark(iterations=0)


def test_benchmark_rejects_negative_iterations():
    import pytest

    with pytest.raises(ValueError):
        run_benchmark(iterations=-5)
