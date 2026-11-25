def validate_guide_sequence(guide_sequence):
    """
    Validate and normalize a CRISPR guide DNA sequence.
    Only A, T, G and C are allowed.
    """
    if not isinstance(guide_sequence, str):
        raise ValueError("Guide sequence must be a string")

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

def validate_gene_records(sequences):
    """
    Validate gene records used for off-target analysis.
    """
    validated_sequences = []

    for index, gene in enumerate(sequences):
        if not isinstance(gene, dict):
            raise ValueError(
                f"Gene record at index {index} must be a dictionary"
            )

        if "id" not in gene:
            raise ValueError(
                f"Gene record at index {index} is missing 'id'"
            )

        if "sequence" not in gene:
            raise ValueError(
                f"Gene record at index {index} is missing 'sequence'"
            )

        if not isinstance(gene["id"], str) or not gene["id"].strip():
            raise ValueError(
                f"Gene ID at index {index} must be a non-empty string"
            )

        if not isinstance(gene["sequence"], str):
            raise ValueError(
                f"Gene sequence at index {index} must be a string"
            )

        validated_gene = gene.copy()
        validated_gene["sequence"] = gene["sequence"].upper()

        validated_sequences.append(validated_gene)

    return validated_sequences
