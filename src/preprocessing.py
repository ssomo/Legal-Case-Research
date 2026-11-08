"""Clean and chunk the selected North Carolina legal opinions."""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from config import RAW_DATA_PATH, PROCESSED_DIR

INPUT_PATH = RAW_DATA_PATH
OUTPUT_PATH = PROCESSED_DIR / "nc_ip_opinion_chunks.csv"

def clean_text(text: str) -> str:
    """Remove extra whitespace from an opinion while keeping its wording."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(
    text: str,
    chunk_words: int = 300,
    overlap_words: int = 50,
) -> list[str]:
    """Split text into word-based chunks with a small overlap."""

    # Validate the chunking parameters
    if chunk_words <= 0:
        raise ValueError("The chunk_words value must be greater than zero.")

    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("The overlap_words value must be between 0 and the chunk_words value.")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap_words

    return chunks


def preprocess_opinions() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find the input file: {INPUT_PATH}"
        )

    opinions = pd.read_csv(INPUT_PATH)
    required_columns = {"case_id", "opinion_index", "opinion_text"}
    missing_columns = required_columns - set(opinions.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    chunk_rows = []
    processed_opinions = 0
    skipped_opinions = 0

    for _, opinion in opinions.iterrows():
        text = opinion["opinion_text"]
        if not isinstance(text, str) or not text.strip():
            skipped_opinions += 1
            continue

        cleaned = clean_text(text)
        chunks = split_into_chunks(cleaned)

        if not chunks:
            skipped_opinions += 1
            continue

        processed_opinions += 1

        # Keep case and opinion details, but don't repeat the full opinion
        # text in every chunk row.
        metadata = opinion.drop(labels=["opinion_text"]).to_dict()

        for chunk_index, chunk_text in enumerate(chunks):
            chunk_rows.append({
                **metadata,
                "chunk_index": chunk_index,
                "chunk_text": chunk_text,
                "chunk_id": f"{opinion['case_id']}_{opinion['opinion_index']}_{chunk_index}"
            })

    chunked_opinions = pd.DataFrame(chunk_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunked_opinions.to_csv(OUTPUT_PATH, index=False)

    print(f"Opinions read: {len(opinions):,}")
    print(f"Opinions processed: {processed_opinions:,}")
    print(f"Opinions skipped: {skipped_opinions:,}")
    print(f"Chunks created: {len(chunked_opinions):,}")
    print(f"Saved to: {OUTPUT_PATH}")

    return chunked_opinions


if __name__ == "__main__":
    preprocess_opinions()
    