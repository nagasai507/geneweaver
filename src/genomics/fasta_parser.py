from Bio import SeqIO


def parse_fasta(file_path):
    """
    Read a FASTA file and validate DNA sequences.
    Allowed bases: A, T, G, C, N.
    """
    sequences = []
    valid_bases = set("ATGCN")

    for record in SeqIO.parse(file_path, "fasta"):
        sequence = str(record.seq).upper()

        invalid_bases = set(sequence) - valid_bases

        if invalid_bases:
            invalid = ", ".join(sorted(invalid_bases))
            raise ValueError(
                f"Gene '{record.id}' contains invalid DNA base(s): {invalid}"
            )

        sequences.append({
            "id": record.id,
            "description": record.description,
            "sequence": sequence,
            "length": len(sequence)
        })

    return sequences


if __name__ == "__main__":
    print("GeneWeaver FASTA Parser Ready!")