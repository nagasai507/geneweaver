import numpy as np
from numba import cuda

from src.alignment.validation import (
    validate_guide_sequence,
    validate_max_mismatches,
    validate_gene_records,
)


@cuda.jit
def _cuda_find_matches(dna, starts, guide, max_mismatches, output):
    """
    Each CUDA thread checks one possible guide-sized DNA window.
    -1 means the window exceeded the mismatch limit.
    """
    idx = cuda.grid(1)

    if idx < starts.size:
        start = starts[idx]
        mismatches = 0

        for j in range(guide.size):
            if dna[start + j] != guide[j]:
                mismatches += 1

            if mismatches > max_mismatches:
                break

        if mismatches <= max_mismatches:
            output[idx] = mismatches
        else:
            output[idx] = -1


def find_off_targets_cuda(
    sequences,
    guide_sequence,
    max_mismatches=2
):
    """
    Find CRISPR off-target matches using NVIDIA CUDA GPU parallelism.

    Returns the same result format as the CPU baseline.
    """

    guide_sequence = validate_guide_sequence(guide_sequence)
    max_mismatches = validate_max_mismatches(max_mismatches)
    sequences = validate_gene_records(sequences)

    guide_length = len(guide_sequence)

    # Prepare DNA data and candidate-window positions.
    all_dna = "".join(gene["sequence"] for gene in sequences)

    start_parts = []
    gene_ranges = []

    dna_offset = 0
    candidate_offset = 0

    for gene in sequences:
        dna = gene["sequence"]
        window_count = max(0, len(dna) - guide_length + 1)

        if window_count:
            starts = np.arange(
                dna_offset,
                dna_offset + window_count,
                dtype=np.int64
            )
            start_parts.append(starts)

        gene_ranges.append(
            (gene, candidate_offset, window_count)
        )

        candidate_offset += window_count
        dna_offset += len(dna)

    # No possible windows means no matches.
    if candidate_offset == 0:
        return []

    if not cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is unavailable. Check your NVIDIA driver "
            "and numba-cuda installation."
        )

    # Encode DNA characters as integer values.
    dna_array = np.fromiter(
        (ord(base) for base in all_dna),
        dtype=np.int32,
        count=len(all_dna)
    )

    guide_array = np.fromiter(
        (ord(base) for base in guide_sequence),
        dtype=np.int32,
        count=guide_length
    )

    starts_array = np.concatenate(start_parts)

    # Transfer data to GPU.
    d_dna = cuda.to_device(dna_array)
    d_starts = cuda.to_device(starts_array)
    d_guide = cuda.to_device(guide_array)

    d_output = cuda.device_array(
        candidate_offset,
        dtype=np.int32
    )

    # Launch enough CUDA threads to cover all candidate windows.
    threads_per_block = 256
    blocks_per_grid = (
        candidate_offset + threads_per_block - 1
    ) // threads_per_block

    _cuda_find_matches[blocks_per_grid, threads_per_block](
        d_dna,
        d_starts,
        d_guide,
        max_mismatches,
        d_output
    )

    cuda.synchronize()

    # Retrieve mismatch counts from GPU.
    mismatch_results = d_output.copy_to_host()

    # Reconstruct results in the same order and format as CPU.
    results = []

    for gene, result_offset, window_count in gene_ranges:
        dna = gene["sequence"]

        for position in range(window_count):
            mismatches = int(
                mismatch_results[result_offset + position]
            )

            if mismatches < 0:
                continue

            target = dna[position:position + guide_length]

            results.append({
                "gene_id": gene["id"],
                "position": position,
                "target": target,
                "mismatches": mismatches,
                "risk": (
                    "HIGH" if mismatches == 0
                    else "MEDIUM" if mismatches == 1
                    else "LOW"
                )
            })

    return results