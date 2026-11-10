import time

from src.genomics.fasta_parser import parse_fasta
from src.alignment.cpu_baseline import find_off_targets

print("\n========== GENEWEAVER ==========")
print("CPU-Based CRISPR Alignment Engine")
print("================================")

sequences = parse_fasta("data/sample.fasta")
guide = "ATGCGTACGTAG"

print("\nGuide Sequence:", guide)
print("Maximum Allowed Mismatches: 2")
print("Genomic Sequences:", len(sequences))

start_time = time.perf_counter()

results = find_off_targets(
    sequences,
    guide,
    max_mismatches=2
)

end_time = time.perf_counter()

print("\n========== ALIGNMENT RESULTS ==========")

for result in results:
    print(f"\nGene ID: {result['gene_id']}")
    print(f"Position: {result['position']}")
    print(f"Target: {result['target']}")
    print(f"Mismatches: {result['mismatches']}")
    print(f"Risk: {result['risk']}")

print("\n========== PERFORMANCE ==========")
print("Total sequences:", len(sequences))
print("Potential matches:", len(results))
print(f"Execution time: {(end_time - start_time) * 1000:.4f} ms")

print("\nGeneWeaver CPU Baseline Completed!")
