
import time
import statistics
import time
from pathlib import Path
from benchmarks.benchmark import create_test_data

from numba import cuda

from src.genomics.fasta_parser import parse_fasta
from src.alignment.cpu_baseline import find_off_targets
from src.alignment.numba_accelerated import find_off_targets_numba
from src.alignment.cuda_accelerated import find_off_targets_cuda
from src.alignment.dask_accelerated import find_off_targets_dask


PROJECT_ROOT = Path(__file__).resolve().parent
FASTA_FILE = PROJECT_ROOT / "data" / "sample.fasta"

GUIDE_SEQUENCE = "ATGCGTACGTAG"
MAX_MISMATCHES = 2


ENGINES = {
    "1": ("CPU Baseline", find_off_targets),
    "2": ("Numba JIT", find_off_targets_numba),
    "3": ("NVIDIA CUDA GPU", find_off_targets_cuda),
    "4": ("Dask Parallel", find_off_targets_dask),
}


def display_results(results, elapsed_time, engine_name, gene_count):

    print("\n========== ALIGNMENT RESULTS ==========")

    if not results:
        print("No potential matches found.")

    for result in results:
        print(f"\nGene ID: {result['gene_id']}")
        print(f"Position: {result['position']}")
        print(f"Target: {result['target']}")
        print(f"Mismatches: {result['mismatches']}")
        print(f"Risk label: {result['risk']}")

    print("\n========== PERFORMANCE ==========")
    print("Engine:", engine_name)
    print("Total sequences:", gene_count)
    print("Potential matches:", len(results))
    print(f"Execution time: {elapsed_time:.4f} ms")


def run_engine(engine_name, function, sequences):

    start_time = time.perf_counter()

    results = function(
        sequences,
        GUIDE_SEQUENCE,
        MAX_MISMATCHES,
    )

    if engine_name == "NVIDIA CUDA GPU":
        cuda.synchronize()

    elapsed_time = (time.perf_counter() - start_time) * 1000

    display_results(
        results,
        elapsed_time,
        engine_name,
        len(sequences),
    )

    return results




def compare_engines(sequences):

    sequences, _ = create_test_data(
        num_genes=3,
        sequence_length=50000,
    )

    print("\nUsing synthetic benchmark dataset:")
    print("Genes:", len(sequences))
    print("Bases per gene:", len(sequences[0]["sequence"]))
    print(
        "Total DNA bases:",
        sum(len(gene["sequence"]) for gene in sequences)
    )

    print("\n========== ENGINE COMPARISON ==========")

    cuda_available = cuda.is_available()

    if cuda_available:
        print("CUDA GPU detected.")
    else:
        print("CUDA unavailable. Skipping GPU comparison.")

    available_engines = [
        (name, function)
        for key, (name, function) in ENGINES.items()
        if key != "3" or cuda_available
    ]

    timings = {}
    results_by_engine = {}

    print("\n========== WARM-UP ==========")

    for engine_name, function in available_engines:

        print(f"Warming up {engine_name}...")

        results = function(
            sequences,
            GUIDE_SEQUENCE,
            MAX_MISMATCHES,
        )

        if engine_name == "NVIDIA CUDA GPU":
            cuda.synchronize()

        results_by_engine[engine_name] = results

    reference = results_by_engine["CPU Baseline"]

    for engine_name, results in results_by_engine.items():
        if results != reference:
            raise RuntimeError(
                f"{engine_name} produced incorrect results."
            )

    print("Warm-up correctness checks: PASSED")

    print("\n========== PERFORMANCE ==========")

    for engine_name, function in available_engines:

        engine_times = []

        print(f"\nMeasuring {engine_name}...")

        for _ in range(3):

            start = time.perf_counter()

            results = function(
                sequences,
                GUIDE_SEQUENCE,
                MAX_MISMATCHES,
            )

            if engine_name == "NVIDIA CUDA GPU":
                cuda.synchronize()

            elapsed = (
                time.perf_counter() - start
            ) * 1000

            engine_times.append(elapsed)

            if results != reference:
                raise RuntimeError(
                    f"{engine_name} produced different results."
                )

        timings[engine_name] = statistics.median(engine_times)

    print("\n========== MEDIAN EXECUTION TIMES ==========")

    for engine_name, median in timings.items():
        print(f"{engine_name}: {median:.4f} ms")

    cpu_time = timings["CPU Baseline"]

    print("\n========== SPEEDUP VS CPU ==========")

    for engine_name, median in timings.items():

        if engine_name == "CPU Baseline":
            continue

        if median < cpu_time:
            print(
                f"{engine_name}: {cpu_time / median:.2f}x faster"
            )
        else:
            print(
                f"{engine_name}: {median / cpu_time:.2f}x slower"
            )

    print("\nPotential matches:", len(reference))
    print("All engine results identical: YES")


def main():

    print("\n======================================")
    print("       GENEWEAVER")
    print("  CRISPR ALIGNMENT ENGINE")
    print("======================================")

    print("\nGuide Sequence:", GUIDE_SEQUENCE)
    print("Maximum Mismatches:", MAX_MISMATCHES)

    print("\nLoading genomic sequences...")

    sequences = parse_fasta(str(FASTA_FILE))

    print("Sequences loaded:", len(sequences))

    print("\n========== SELECT ENGINE ==========")
    print("1. CPU Baseline")
    print("2. Numba JIT")
    print("3. NVIDIA CUDA GPU")
    print("4. Dask Parallel")
    print("5. Compare All Available Engines")
    print("0. Exit")

    choice = input("\nSelect an option: ").strip()

    if choice == "0":
        print("GeneWeaver closed.")
        return

    if choice == "5":
        compare_engines(sequences)
        return

    if choice not in ENGINES:
        print("Invalid selection. Please choose 0-5.")
        return

    engine_name, function = ENGINES[choice]

    if choice == "3" and not cuda.is_available():
        print("\nCUDA GPU is unavailable.")
        print("Check your NVIDIA driver and CUDA installation.")
        return

    print(f"\nSelected engine: {engine_name}")

    run_engine(
        engine_name,
        function,
        sequences,
    )

    print("\nGeneWeaver execution completed!")


if __name__ == "__main__":
    main()
