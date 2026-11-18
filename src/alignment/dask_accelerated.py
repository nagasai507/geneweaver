
from dask import delayed, compute

from src.alignment.cpu_baseline import find_off_targets


def _process_gene(gene, guide_sequence, max_mismatches):
    """
    Process one genomic sequence independently.
    """
    return find_off_targets(
        [gene],
        guide_sequence,
        max_mismatches
    )


def find_off_targets_dask(
    sequences,
    guide_sequence,
    max_mismatches=2
):
    """
    Find potential off-target matches using Dask parallel processing.
    """

    if not guide_sequence:
        raise ValueError("Guide sequence cannot be empty")

    if max_mismatches < 0:
        raise ValueError("Maximum mismatches cannot be negative")

    tasks = [
        delayed(_process_gene)(
            gene,
            guide_sequence,
            max_mismatches
        )
        for gene in sequences
    ]

    results = compute(*tasks, scheduler="threads")

    return [
        match
        for gene_results in results
        for match in gene_results
    ]
