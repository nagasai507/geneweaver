
import sys
import time
import random
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from numba import cuda

from src.alignment.cpu_baseline import find_off_targets
from src.alignment.numba_accelerated import find_off_targets_numba
from src.alignment.dask_accelerated import find_off_targets_dask
from src.alignment.cuda_accelerated import find_off_targets_cuda


def _validate_matching_results(
    cpu_results,
    numba_results,
    dask_results,
    cuda_results=None,
):
    """Ensure all available engines produce identical results."""

    if not (cpu_results == numba_results == dask_results):
        raise RuntimeError(
            "CPU, Numba, and Dask produced different results"
        )

    if cuda_results is not None and cpu_results != cuda_results:
        raise RuntimeError(
            "CUDA produced different results"
        )


def _validate_positive_integer(value, parameter_name):
    """Validate a positive integer. Booleans are not accepted."""

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
    """Generate reproducible synthetic DNA sequences."""

    rng = random.Random(42)
    guide = "ATGCGTACGTAG"
    sequences = []

    num_genes = _validate_positive_integer(
        num_genes, "Number of genes"
    )

    sequence_length = _validate_positive_integer(
        sequence_length, "Sequence length"
    )

    if sequence_length < 25012:
        raise ValueError("Sequence length is too short")

    for i in range(num_genes):
        dna = [
            rng.choice("ATGC")
            for _ in range(sequence_length)
        ]

        # Insert two known exact matches.
        dna[1000:1012] = guide
        dna[25000:25012] = guide

        sequences.append({
            "id": f"Synthetic_Gene_{i + 1}",
            "sequence": "".join(dna),
        })

    return sequences, guide


def run_benchmark(iterations=20):

    iterations = _validate_positive_integer(
        iterations, "Number of iterations"
    )

    dataset_sizes = [
        ("Small", 3),
        ("Medium", 10),
        ("Large", 25),
    ]

    cuda_available = cuda.is_available()

    print("\n==========================================")
    print("       GENEWEAVER PERFORMANCE TEST")
    print("==========================================")

    print("Iterations per engine:", iterations)

    if cuda_available:
        print("CUDA GPU:", cuda.get_current_device().name)
    else:
        print("CUDA GPU: Not available (CUDA test skipped)")

    for dataset_name, num_genes in dataset_sizes:

        sequences, guide = create_test_data(
            num_genes=num_genes,
            sequence_length=50000,
        )

        total_bases = sum(
            len(gene["sequence"])
            for gene in sequences
        )

        print(f"\n{'=' * 45}")
        print(f"{dataset_name.upper()} DATASET")
        print("=" * 45)

        print("Genes:", num_genes)
        print("Bases per gene:", len(sequences[0]["sequence"]))
        print("Total DNA bases:", total_bases)

        # CPU reference.
        print("\nPreparing reference results...")
        reference = find_off_targets(sequences, guide, 2)

        # Prepare available engines.
        methods = {
            "CPU": find_off_targets,
            "Numba": find_off_targets_numba,
            "Dask": find_off_targets_dask,
        }

        if cuda_available:
            methods["CUDA"] = find_off_targets_cuda

        # Warm up all engines before measurement.
        print("Warming up engines...")

        warmup_results = {}

        for name, function in methods.items():
            result = function(sequences, guide, 2)

            if name == "CUDA":
                cuda.synchronize()

            warmup_results[name] = result

        _validate_matching_results(
            reference,
            warmup_results["Numba"],
            warmup_results["Dask"],
            warmup_results.get("CUDA"),
        )

        print("Warm-up and correctness checks: PASSED")

        # Measure each engine.
        timings = {}
        final_results = {}

        for name, function in methods.items():

            print(f"\nBenchmarking {name}...")

            engine_times = []

            for iteration in range(iterations):

                start = time.perf_counter()

                result = function(sequences, guide, 2)

                if name == "CUDA":
                    cuda.synchronize()

                elapsed = (
                    time.perf_counter() - start
                ) * 1000

                engine_times.append(elapsed)
                final_results[name] = result

                if result != reference:
                    raise RuntimeError(
                        f"{name} results differ from CPU reference"
                    )

            timings[name] = engine_times

        # Validate final outputs.
        _validate_matching_results(
            final_results["CPU"],
            final_results["Numba"],
            final_results["Dask"],
            final_results.get("CUDA"),
        )

        # Display performance results.
        print("\n---------- RESULTS ----------")

        medians = {}

        for name, engine_times in timings.items():

            average = statistics.mean(engine_times)
            median = statistics.median(engine_times)

            medians[name] = median

            print(f"\n{name}:")
            print(f"  Average: {average:.2f} ms")
            print(f"  Median:  {median:.2f} ms")

        # Compare median execution times.
        cpu_median = medians["CPU"]

        print("\n------ SPEEDUP VS CPU ------")

        for name, median in medians.items():

            if name == "CPU":
                continue

            if median < cpu_median:
                print(
                    f"{name}: "
                    f"{cpu_median / median:.2f}x faster"
                )
            else:
                print(
                    f"{name}: "
                    f"{median / cpu_median:.2f}x slower"
                )

        print("\nPotential matches:", len(reference))
        print("All engine results identical: YES")

    print("\n==========================================")
    print("  GENEWEAVER BENCHMARK COMPLETED")
    print("==========================================")


if __name__ == "__main__":
    run_benchmark()
