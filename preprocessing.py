"""Clean and chunk the selected North Carolina legal opinions."""

from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "nc_ip_opinions.csv"
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "nc_ip_opinion_chunks.csv"
)


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
    if "opinion_text" not in opinions.columns:
        raise ValueError("The input CSV must contain an opinion_text column.")

    chunk_rows = []

    for _, opinion in opinions.iterrows():
        text = opinion["opinion_text"]
        if not isinstance(text, str) or not text.strip():
            continue

        cleaned = clean_text(text)
        chunks = split_into_chunks(cleaned)

        # Keep case and opinion details, but don't repeat the full opinion
        # text in every chunk row.
        metadata = opinion.drop(labels=["opinion_text"]).to_dict()

        for chunk_index, chunk_text in enumerate(chunks):
            chunk_rows.append({
                **metadata,
                "chunk_index": chunk_index,
                "chunk_text": chunk_text,
            })

    chunked_opinions = pd.DataFrame(chunk_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunked_opinions.to_csv(OUTPUT_PATH, index=False)

    print(f"Opinions read: {len(opinions):,}")
    print(f"Chunks created: {len(chunked_opinions):,}")
    print(f"Saved to: {OUTPUT_PATH}")

    return chunked_opinions


if __name__ == "__main__":
    preprocess_opinions()
    