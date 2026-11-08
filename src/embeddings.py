"""Create and store embeddings for the preprocessed opinion chunks."""
from __future__ import annotations

import chromadb
import ollama
import pandas as pd

from config import CHROMA_DIR, PROCESSED_DATA_PATH, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, COLLECTION_NAME

INPUT_PATH = PROCESSED_DATA_PATH

def to_chroma_metadata(row: dict) -> dict:
    """Keep metadata values in types ChromaDB can store."""
    metadata = {}

    for key, value in row.items():
        if key == "chunk_text":
            continue

        if hasattr(value, "item"):
            value = value.item()

        if value is None or pd.isna(value):
            continue

        if isinstance(value, (str, int, float, bool)):
            metadata[str(key)] = value
        else:
            metadata[str(key)] = str(value)

    return metadata


def create_embeddings() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Could not find: {INPUT_PATH}")

    chunks = pd.read_csv(INPUT_PATH)

    # Validate required columns
    required_columns = {
        "chunk_id",
        "case_id",
        "opinion_index",
        "chunk_index",
        "chunk_text",
    }

    missing_columns = required_columns - set(chunks.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    # Remove empty chunks
    chunks = chunks.dropna(subset=["chunk_text"]).copy()
    chunks["chunk_text"] = chunks["chunk_text"].astype(str).str.strip()
    chunks = chunks[chunks["chunk_text"] != ""]

    # Validate chunks
    if chunks.empty:
        raise ValueError("No valid opinion chunks were found.")

    if chunks["chunk_id"].duplicated().any():
        raise ValueError("Duplicate chunk IDs were found.")

    # Initialize the Ollama client
    ollama_client = ollama.Client(host=OLLAMA_BASE_URL)

    # Initialize the ChromaDB client
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 50

    # Process chunks in batches of 50
    for start in range(0, len(chunks), batch_size):
        batch = chunks.iloc[start : start + batch_size]
        documents = batch["chunk_text"].tolist()

        ids = batch["chunk_id"].astype(str).tolist()
        metadatas = [to_chroma_metadata(row) for row in batch.to_dict(orient="records")]

        # Use ollama to generate embeddings
        response = ollama_client.embed(
            model=OLLAMA_EMBED_MODEL,
            input=documents
        )

        embeddings = response.embeddings

        # Store the embeddings, documents, and metadata in the ChromaDB collection
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        print(f"Processed {min(start + len(batch), len(chunks)):,} of {len(chunks):,} chunks")

    print(f"Embedding model: {OLLAMA_EMBED_MODEL}")
    print(f"Embedding collection: {COLLECTION_NAME}")
    print(f"Total chunks stored: {collection.count():,}")
    print(f"Saved ChromaDB files to: {CHROMA_DIR}")


if __name__ == "__main__":
    create_embeddings()