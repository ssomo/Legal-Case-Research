"""Create and store embeddings for the preprocessed opinion chunks."""
from __future__ import annotations

import chromadb
import ollama
import pandas as pd

from config import PROCESSED_DIR, CHROMA_DIR, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, COLLECTION_NAME

INPUT_PATH = PROCESSED_DIR / "nc_ip_opinion_chunks.csv"

EMBEDDING_MODEL = OLLAMA_EMBED_MODEL


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
    """Generate and store embeddings for opinion chunks."""

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
    chunks = chunks.dropna(subset=['chunk_text']).copy()
    chunks["chunk_text"] = chunks["chunk_text"].astype(str).str.strip()
    chunks = chunks[chunks["chunk_text"] != ""]

    # Validation
    if chunks.empty:
        raise ValueError("No valid opinion chunks were found.")

    if chunks["chunk_id"].duplicated().any():
        raise ValueError("Duplicate chunk IDs were found.")

    # Initialize Ollama client
    ollama_client = ollama.Client(host=OLLAMA_BASE_URL)

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 50

    # Process the chunks in batches
    for start in range(0, len(chunks), batch_size):
        batch = chunks.iloc[start : start + batch_size]
        documents = batch["chunk_text"].tolist()

        # Reuse chunk IDs from preprocessing.py
        ids = batch["chunk_id"].astype(str).tolist()
        metadatas = [to_chroma_metadata(row) for row in batch.to_dict(orient="records")]

        # Generate text embeddings using Ollama
        response = ollama_client.embed(
            model=EMBEDDING_MODEL,
            input=documents
        )

        embeddings = response.embeddings

        # Store the documents, embeddings, and metadata in the Chroma DB
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        print(f"Processed {min(start + len(batch), len(chunks)):,} of {len(chunks):,} chunks")

    print(f"\nEmbedding Model: {EMBEDDING_MODEL}")
    print(f"Embedding collection: {COLLECTION_NAME}")
    print(f"Total chunks stored: {collection.count():,}")
    print(f"Saved ChromaDB files to: {CHROMA_DIR}")


if __name__ == "__main__":
    create_embeddings()