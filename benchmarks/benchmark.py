import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import time
import statistics

from src.genomics.fasta_parser import parse_fasta
from src.alignment.cpu_baseline import find_off_targets


def run_benchmark(iterations=1000):

    sequences = parse_fasta("data/sample.fasta")
    guide = "ATGCGTACGTAG"

    timings = []

    print("\n===== GENEWEAVER CPU BENCHMARK =====")
    print("Iterations:", iterations)

    # Warm-up
    find_off_targets(sequences, guide)

    for _ in range(iterations):
        start = time.perf_counter()

        results = find_off_targets(
            sequences,
            guide,
            max_mismatches=2
        )

        end = time.perf_counter()
        timings.append((end - start) * 1000)

    print("\n===== BENCHMARK RESULTS =====")
    print("Sequences:", len(sequences))
    print("Matches:", len(results))
    print(f"Average: {statistics.mean(timings):.6f} ms")
    print(f"Median: {statistics.median(timings):.6f} ms")
    print(f"Minimum: {min(timings):.6f} ms")
    print(f"Maximum: {max(timings):.6f} ms")

    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    run_benchmark()
