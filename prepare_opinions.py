"""Create an opinion-level North Carolina case dataset."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

from datasets import load_dataset

from config import RAW_DATA_PATH


KEYWORDS = [
    "intellectual property",
    "copyright infringement",
    "trademark",
    "patent",
    "trade secret",
    "intellectual property rights",
    "fair use",
    "copyright owner",
    "copyrighted work",
    "infringement",
    "unauthorized reproduction",
    "derivative work",
    "right of publicity",
    "DMCA",
]

COLUMNS = [
    "case_id",
    "case_name",
    "date_filed",
    "court_jurisdiction",
    "court_full_name",
    "nature_of_suit",
    "topic_matches",
    "opinion_index",
    "opinion_author",
    "opinion_type",
    "opinion_text",
]


def main() -> None:
    token = os.getenv("HF_ACCESS_TOKEN")
    dataset = load_dataset("harvard-lil/cold-cases", token=token)
    train = dataset["train"]

    output_path = Path(RAW_DATA_PATH).with_name("nc_ip_opinions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name("nc_ip_opinions.tmp.csv")

    pattern = re.compile(
        "|".join(re.escape(word) for word in sorted(KEYWORDS, key=len, reverse=True)),
        flags=re.IGNORECASE,
    )

    scanned = 0
    nc_cases = 0
    matching_cases = 0
    saved_opinions = 0
    blank_texts = 0

    with temp_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
        writer.writeheader()

        for scanned, case in enumerate(train, start=1):
            if scanned % 250_000 == 0:
                print(f"Scanned {scanned:,} cases...", flush=True)

            if case.get("court_jurisdiction") != "North Carolina, NC":
                continue

            nc_cases += 1
            opinions = case.get("opinions") or []

            if not isinstance(opinions, (list, tuple)):
                try:
                    opinions = list(opinions)
                except TypeError:
                    continue

            valid_opinions = [
                opinion for opinion in opinions if isinstance(opinion, dict)
            ]

            matched_terms = set()
            for opinion in valid_opinions:
                text = opinion.get("opinion_text")
                if isinstance(text, str):
                    matched_terms.update(
                        match.group(0).lower()
                        for match in pattern.finditer(text)
                    )

            if not matched_terms:
                continue

            matching_cases += 1
            topic_matches = ", ".join(sorted(matched_terms))

            for index, opinion in enumerate(opinions):
                if not isinstance(opinion, dict):
                    continue

                text = opinion.get("opinion_text")
                if not isinstance(text, str):
                    text = ""
                if not text.strip():
                    blank_texts += 1

                writer.writerow({
                    "case_id": case.get("id"),
                    "case_name": case.get("case_name"),
                    "date_filed": case.get("date_filed"),
                    "court_jurisdiction": case.get("court_jurisdiction"),
                    "court_full_name": case.get("court_full_name"),
                    "nature_of_suit": case.get("nature_of_suit"),
                    "topic_matches": topic_matches,
                    "opinion_index": index,
                    "opinion_author": opinion.get("author_str"),
                    "opinion_type": opinion.get("type"),
                    "opinion_text": text,
                })
                saved_opinions += 1

    temp_path.replace(output_path)

    print(f"Rows scanned: {scanned:,}")
    print(f"North Carolina cases: {nc_cases:,}")
    print(f"Cases matching keywords in any opinion: {matching_cases:,}")
    print(f"Opinion records saved: {saved_opinions:,}")
    print(f"Opinions with blank text: {blank_texts:,}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
    