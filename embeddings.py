"""Create and store embeddings for the preprocessed opinion chunks."""

from pathlib import Path

import chromadb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "nc_ip_opinion_chunks.csv"
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "nc_ip_opinion_chunks"


def to_chroma_metadata(row: dict) -> dict:
    """Keep metadata values in types ChromaDB can store."""
    metadata = {}

    for key, value in row.items():
        if key == "chunk_text" or pd.isna(value):
            continue

        if hasattr(value, "item"):
            value = value.item()

        if isinstance(value, (str, int, float, bool)):
            metadata[str(key)] = value
        else:
            metadata[str(key)] = str(value)

    return metadata


def create_embeddings() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Could not find: {INPUT_PATH}")

    chunks = pd.read_csv(INPUT_PATH)
    if "chunk_text" not in chunks.columns:
        raise ValueError("The CSV must contain a chunk_text column.")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 250

    for start in range(0, len(chunks), batch_size):
        batch = chunks.iloc[start : start + batch_size]
        documents = batch["chunk_text"].fillna("").astype(str).tolist()

        ids = []
        metadatas = []

        for row_number, (_, row) in enumerate(batch.iterrows()):
            case_id = row.get("case_id", row.get("id", start + row_number))
            opinion_index = row.get("opinion_index", 0)
            chunk_index = row.get("chunk_index", start + row_number)

            ids.append(f"{case_id}_{opinion_index}_{chunk_index}")

            row_metadata = row.to_dict()
            row_metadata["case_id"] = str(case_id)
            metadatas.append(to_chroma_metadata(row_metadata))

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        print(f"Processed {min(start + len(batch), len(chunks)):,} of {len(chunks):,} chunks")

    print(f"Embedding collection: {COLLECTION_NAME}")
    print(f"Saved ChromaDB files to: {CHROMA_PATH}")


if __name__ == "__main__":
    create_embeddings()