"""
Pipeline for semantic retrieval of relevant documents from a vector database.

Converts a query into a vector representation and retrieves the most relevant documents from the database based on similarity.
"""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config import CHROMA_DIR, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, COLLECTION_NAME

def load_vector_store() -> Chroma:
    """
    Load the existing ChromaDB vector store.
    """
    embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    
    return vector_store

def retrieve_relevant_chunks(query: str, vector_store: Chroma, k: int = 5) -> list:
    """Retrieve the top-k relevant opinion chunks using semantic similarity."""
    # Validate the retrieval parameters
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    # Retrieve more chunks to account for duplicate cases
    results = vector_store.similarity_search_with_relevance_scores(
        query=query.strip(),
        k=20
    )

    # Keep only the highest-scoring chunk from each unique case
    unique_cases = []
    seen_cases = set()

    for document, score in results:
        case_id = document.metadata.get("case_id")

        if case_id not in seen_cases:
            seen_cases.add(case_id)
            unique_cases.append((document, score))

        if len(unique_cases) == k:
            break

    return unique_cases
