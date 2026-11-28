IUPAC_BASES = {
    "A": set("A"),
    "T": set("T"),
    "G": set("G"),
    "C": set("C"),
    "N": set("ATGCN"),
}


def is_valid_pam(pam_sequence, pam_pattern="NGG"):
    """Check whether a PAM sequence matches a configurable IUPAC pattern."""
    if not isinstance(pam_sequence, str) or not isinstance(pam_pattern, str):
        raise ValueError("PAM sequence and pattern must be strings")

    pam_sequence = pam_sequence.upper()
    pam_pattern = pam_pattern.upper()

    if not pam_pattern:
        raise ValueError("PAM pattern cannot be empty")

    invalid_codes = set(pam_pattern) - set(IUPAC_BASES)
    if invalid_codes:
        raise ValueError(
            f"PAM pattern contains unsupported code(s): {', '.join(sorted(invalid_codes))}"
        )

    if len(pam_sequence) != len(pam_pattern):
        return False

    if set(pam_sequence) - set("ATGCN"):
        return False

    return all(
        base in IUPAC_BASES[code]
        for base, code in zip(pam_sequence, pam_pattern)
    )


def reverse_complement(sequence):
    """Return the reverse complement of a DNA sequence."""
    if not isinstance(sequence, str):
        raise ValueError("DNA sequence must be a string")

    sequence = sequence.upper()

    if set(sequence) - set("ATGCN"):
        raise ValueError("Invalid DNA sequence")

    return sequence.translate(
        str.maketrans("ATGCN", "TACGN")
    )[::-1]


def find_pam_sites(sequence, pam_pattern="NGG", protospacer_length=20):
    """
    Find PAM sites on both DNA strands.

    Coordinates are zero-based and end-exclusive.
    """
    if not isinstance(sequence, str):
        raise ValueError("DNA sequence must be a string")

    if not isinstance(protospacer_length, int) or isinstance(protospacer_length, bool):
        raise ValueError("Protospacer length must be an integer")

    if protospacer_length <= 0:
        raise ValueError("Protospacer length must be positive")

    sequence = sequence.upper()

    if set(sequence) - set("ATGCN"):
        raise ValueError("Invalid DNA sequence")

    if not isinstance(pam_pattern, str) or not pam_pattern:
        raise ValueError("PAM pattern must be a non-empty string")

    if set(pam_pattern.upper()) - set(IUPAC_BASES):
        raise ValueError("Invalid PAM pattern")

    pam_pattern = pam_pattern.upper()
    pam_length = len(pam_pattern)

    results = []

    # Forward strand: protospacer followed by PAM
    for i in range(protospacer_length, len(sequence) - pam_length + 1):
        pam_start = i
        pam_end = i + pam_length

        pam = sequence[pam_start:pam_end]

        if is_valid_pam(pam, pam_pattern):
            results.append({
                "strand": "+",
                "protospacer": sequence[i - protospacer_length:i],
                "pam": pam,
                "start": i - protospacer_length,
                "end": pam_end,
            })

    # Reverse strand: PAM followed by protospacer
    for i in range(0, len(sequence) - pam_length - protospacer_length + 1):
        pam = sequence[i:i + pam_length]

        if is_valid_pam(pam, reverse_complement(pam_pattern)):
            protospacer_start = i + pam_length
            protospacer_end = protospacer_start + protospacer_length

            results.append({
                "strand": "-",
                "protospacer": reverse_complement(
                    sequence[protospacer_start:protospacer_end]
                ),
                "pam": reverse_complement(pam),
                "start": i,
                "end": protospacer_end,
            })

    return sorted(results, key=lambda site: site["start"])