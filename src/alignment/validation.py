def validate_guide_sequence(guide_sequence):
    """
    Validate and normalize a CRISPR guide DNA sequence.
    Only A, T, G and C are allowed.
    """
    if not guide_sequence:
        raise ValueError("Guide sequence cannot be empty")

    guide_sequence = guide_sequence.upper()

    invalid_bases = set(guide_sequence) - set("ATGC")

    if invalid_bases:
        invalid = ", ".join(sorted(invalid_bases))
        raise ValueError(
            f"Guide sequence contains invalid DNA base(s): {invalid}"
        )

    return guide_sequence


def validate_max_mismatches(max_mismatches):
    """
    Validate the maximum allowed number of mismatches.
    It must be a non-negative integer.
    """
    if (
        isinstance(max_mismatches, bool)
        or not isinstance(max_mismatches, int)
        or max_mismatches < 0
    ):
        raise ValueError(
            "Maximum mismatches must be a non-negative integer"
        )

    return max_mismatches
