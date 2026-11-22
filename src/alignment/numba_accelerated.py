from numba import njit
from src.alignment.validation import validate_guide_sequence

@njit
def _find_matches(gene_sequences, guide_sequence, max_mismatches):
    matches = []
    guide_length = len(guide_sequence)

    for gene_index in range(len(gene_sequences)):
        dna = gene_sequences[gene_index]

        for position in range(len(dna) - guide_length + 1):
            mismatches = 0

            for i in range(guide_length):
                if guide_sequence[i] != dna[position + i]:
                    mismatches += 1

            if mismatches <= max_mismatches:
                matches.append((gene_index, position, mismatches))

    return matches

def find_off_targets_numba(sequences, guide_sequence, max_mismatches=2):
    guide_sequence = validate_guide_sequence(guide_sequence)

    if max_mismatches < 0:
        raise ValueError("Maximum mismatches cannot be negative")

    gene_sequences = [
        gene["sequence"].upper() for gene in sequences
    ]


    raw_matches = _find_matches(
    tuple(gene_sequences),
    guide_sequence,
    max_mismatches
    )

    results = []

    for gene_index, position, mismatches in raw_matches:
        target = gene_sequences[gene_index][
            position:position + len(guide_sequence)
        ]

        results.append({
            "gene_id": sequences[gene_index]["id"],
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