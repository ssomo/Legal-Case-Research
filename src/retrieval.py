"""
Pipeline for semantic retrieval of relevant documents from a vector database.

Converts a query into a vector representation and retrieves the most relevant documents from the database based on similarity.
"""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from config import CHROMA_DIR, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, COLLECTION_NAME

from nltk.tokenize import sent_tokenize

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

def highlight_relevant_sentences(query: str, document: str, embeddings, top_n: int = 3) -> list:
    """Find the sentences most relevant to a user's query."""
    # Validate the parameters
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if not isinstance(document, str) or not document.strip():
        return []

    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
        raise ValueError("top_n must be a positive integer.")
    
    # Split the document into sentences
    sentences = sent_tokenize(document)

    # Remove short sentence fragments
    sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) >= 8]

    if not sentences:
        return []

    # Convert the sentences into LangChain documents
    sentence_docs = [Document(page_content=sentence, metadata={"index": i}) for i, sentence in enumerate(sentences)]

    # Create an in-memory vector store
    vector_store = InMemoryVectorStore.from_documents(
        documents=sentence_docs,
        embedding=embeddings
    )

    # Retrieve the most relevant sentences
    results = vector_store.similarity_search_with_score(
        query=query,
        k=min(top_n, len(sentence_docs))
    )

    return results

def retrieve_full_opinion(case_id: str, vector_store: Chroma) -> str:
    """Retrieve and reconstruct the full opinion from its chunks."""
    # Retrieve all chunks associated with a case
    results = vector_store._collection.get(
        where={"case_id": case_id},
        include=["documents", 'metadatas']
    )

    documents = results['documents']
    metadatas = results['metadatas']

    if not documents:
        return ""

    # Combine chunks in their original order
    chunks = sorted(zip(documents, metadatas), key=lambda x: x[1].get("chunk_index", 0))

    full_opinion = "\n".join(text for text, metadata in chunks)

    return full_opinion