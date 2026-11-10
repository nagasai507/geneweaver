
from Bio import SeqIO


def parse_fasta(file_path):
    """
    Read a FASTA file and extract genomic sequences.
    """
    sequences = []

    for record in SeqIO.parse(file_path, "fasta"):
        sequence = str(record.seq).upper()

        # Keep only valid DNA bases
        sequence = "".join(
            base for base in sequence if base in "ATGC"
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
