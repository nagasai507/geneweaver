
import sys
import time
import random
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.cpu_baseline import find_off_targets
from src.alignment.numba_accelerated import find_off_targets_numba
from src.alignment.dask_accelerated import find_off_targets_dask

def _validate_positive_integer(value, parameter_name):
    """
    Validate that a value is a positive integer.
    Boolean values are not accepted as integers.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{parameter_name} must be a positive integer"
        )
    return value


def create_test_data(num_genes=3, sequence_length=50000):
    rng = random.Random(42)
    guide = "ATGCGTACGTAG"
    sequences = []

    num_genes = _validate_positive_integer(num_genes, "Number of genes")
    sequence_length = _validate_positive_integer(
        sequence_length, "Sequence length"
    )

    if sequence_length < 25012:
        raise ValueError("Sequence length is too short")

    for i in range(num_genes):
        dna = [rng.choice("ATGC") for _ in range(sequence_length)]

        # Insert known exact matches
        dna[1000:1012] = guide
        dna[25000:25012] = guide

        sequences.append({
            "id": f"Synthetic_Gene_{i + 1}",
            "sequence": "".join(dna)
        })

    return sequences, guide


def run_benchmark(iterations=20):

    iterations = _validate_positive_integer(
        iterations, "Number of iterations"
    )

    dataset_sizes = [
        ("Small", 3),
        ("Medium", 10),
        ("Large", 25)
    ]

    print("\n===== GENEWEAVER SCALABILITY BENCHMARK =====")
    print("Iterations per method:", iterations)

    for dataset_name, num_genes in dataset_sizes:

        sequences, guide = create_test_data(
            num_genes=num_genes,
            sequence_length=50000
        )

        total_bases = sum(
            len(gene["sequence"]) for gene in sequences
        )

        print(f"\n===== {dataset_name.upper()} DATASET =====")
        print("Number of genes:", num_genes)
        print("Bases per gene:", len(sequences[0]["sequence"]))
        print("Total DNA bases:", total_bases)

        # Warm up Numba and Dask before measuring
        find_off_targets_numba(sequences, guide, 2)
        find_off_targets_dask(sequences, guide, 2)

        cpu_times = []
        numba_times = []
        dask_times = []

        for _ in range(iterations):

            start = time.perf_counter()
            cpu_results = find_off_targets(sequences, guide, 2)
            cpu_times.append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            numba_results = find_off_targets_numba(sequences, guide, 2)
            numba_times.append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            dask_results = find_off_targets_dask(sequences, guide, 2)
            dask_times.append((time.perf_counter() - start) * 1000)

        # Verify that all methods produce identical results
        assert cpu_results == numba_results == dask_results

        cpu_average = statistics.mean(cpu_times)
        numba_average = statistics.mean(numba_times)
        dask_average = statistics.mean(dask_times)

        print("\nCPU average:", f"{cpu_average:.4f} ms")
        print("CPU median:", f"{statistics.median(cpu_times):.4f} ms")

        print("\nNumba average:", f"{numba_average:.4f} ms")
        print("Numba median:", f"{statistics.median(numba_times):.4f} ms")

        print("\nDask average:", f"{dask_average:.4f} ms")
        print("Dask median:", f"{statistics.median(dask_times):.4f} ms")

        print("\nPotential matches:", len(cpu_results))
        print("Results identical: YES")

        print("\nPerformance compared with CPU:")

        for name, average in [
            ("Numba", numba_average),
            ("Dask", dask_average)
        ]:
            if average < cpu_average:
                print(f"{name} speedup: {cpu_average / average:.2f}x")
            else:
                print(
                    f"{name} performance ratio: "
                    f"{average / cpu_average:.2f}x slower"
                )

    print("\n===== SCALABILITY BENCHMARK COMPLETED =====")



if __name__ == "__main__":
    run_benchmark()
