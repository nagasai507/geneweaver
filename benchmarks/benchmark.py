
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


def create_test_data():
    rng = random.Random(42)
    guide = "ATGCGTACGTAG"
    sequences = []

    for i in range(3):
        dna = [rng.choice("ATGC") for _ in range(50000)]

        # Insert known exact matches
        dna[1000:1012] = guide
        dna[25000:25012] = guide

        sequences.append({
            "id": f"Synthetic_Gene_{i + 1}",
            "sequence": "".join(dna)
        })

    return sequences, guide


def run_benchmark(iterations=20):

    sequences, guide = create_test_data()

    print("\n===== GENEWEAVER LARGE DATA BENCHMARK =====")
    print("Synthetic DNA sequences:", len(sequences))
    print("Bases per sequence:", len(sequences[0]["sequence"]))
    print("Total DNA bases:", sum(len(g["sequence"]) for g in sequences))
    print("Iterations:", iterations)

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

    # Verify that all three methods produce identical results
    assert cpu_results == numba_results == dask_results

    cpu_average = statistics.mean(cpu_times)
    numba_average = statistics.mean(numba_times)
    dask_average = statistics.mean(dask_times)

    print("\n===== CPU RESULTS =====")
    print(f"Average: {cpu_average:.4f} ms")
    print(f"Median: {statistics.median(cpu_times):.4f} ms")

    print("\n===== NUMBA RESULTS =====")
    print(f"Average: {numba_average:.4f} ms")
    print(f"Median: {statistics.median(numba_times):.4f} ms")

    print("\n===== DASK RESULTS =====")
    print(f"Average: {dask_average:.4f} ms")
    print(f"Median: {statistics.median(dask_times):.4f} ms")

    print("\n===== PERFORMANCE SUMMARY =====")
    print("Potential matches:", len(cpu_results))
    print("Results identical: YES")

    for name, average in [
        ("Numba", numba_average),
        ("Dask", dask_average)
    ]:
        if average < cpu_average:
            print(f"{name} speedup: {cpu_average / average:.2f}x")
        else:
            print(f"{name} performance ratio: {average / cpu_average:.2f}x slower")

    print("\nThree-method benchmark completed!")


if __name__ == "__main__":
    run_benchmark()
