from dask import delayed, compute

from src.alignment.cpu_baseline import find_off_targets
from src.alignment.validation import validate_guide_sequence


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

    guide_sequence = validate_guide_sequence(guide_sequence)

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
