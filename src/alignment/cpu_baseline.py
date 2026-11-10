def find_off_targets(sequences, guide_sequence, max_mismatches=2):
    """
    Find DNA regions matching a CRISPR guide sequence.
    Uses sliding-window Hamming distance.
    """

    guide_sequence = guide_sequence.upper()
    guide_length = len(guide_sequence)

    results = []

    for gene in sequences:
        dna = gene["sequence"]

        for position in range(len(dna) - guide_length + 1):
            target = dna[position:position + guide_length]

            mismatches = sum(
                a != b for a, b in zip(guide_sequence, target)
            )

            if mismatches <= max_mismatches:
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
