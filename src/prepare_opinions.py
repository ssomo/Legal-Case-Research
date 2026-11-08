"""Create an opinion-level North Carolina case dataset."""
from __future__ import annotations

import csv
import os
import re

from datasets import load_dataset

from config import RAW_DATA_PATH

    
KEYWORDS = [
    # General intellectual property
    "intellectual property",
    "intellectual property rights",
    "intellectual property infringement",

    # Copyright
    "copyright",
    "copyright infringement",
    "copyright ownership",
    "copyright registration",
    "copyright protection",
    "copyright holder",
    "copyrighted work",
    "unauthorized copying",
    "unauthorized reproduction",
    "derivative work",
    "fair use",
    "public domain",
    "digital millennium copyright",
    "DMCA",

    # Trademarks and unfair competition
    "trademark",
    "service mark",
    "trademark infringement",
    "trade dress",
    "likelihood of confusion",
    "false designation of origin",
    "Lanham Act",
    "unfair competition",
    "trademark dilution",
    "trademark infringement claim",

    # Patents
    "patent",
    "patent infringement",
    "patent ownership",
    "patent validity",
    "patent claim",
    "patent rights",

    # Trade secrets
    "trade secret",
    "trade secret misappropriation",
    "misappropriation of trade secrets",
    "confidential information",
    "proprietary information",
    "trade secret protection",

    # Licensing and ownership
    "intellectual property license",
    "copyright license",
    "patent license",
    "trademark license",
    "licensing agreement",
    "royalty",
    "ownership of intellectual property",
    "right of publicity",
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

    output_path = RAW_DATA_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name("nc_ip_opinions.tmp.csv")

    # Compile keyword matching pattern
    pattern = re.compile(
        r"\b(?:"
        + "|".join(
            re.escape(word)
            for word in sorted(KEYWORDS, key=len, reverse=True)
        )
        + r")\b",
        flags=re.IGNORECASE,
    )

    scanned = 0
    nc_cases = 0
    matching_cases = 0
    saved_opinions = 0
    blank_texts = 0
    blank_texts_in_matching_cases = 0

    with temp_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
        writer.writeheader()

        for scanned, case in enumerate(train, start=1):
            if scanned % 250_000 == 0:
                print(f"Scanned {scanned:,} cases...", flush=True)

            # Filter NC cases
            if case.get("court_jurisdiction") != "North Carolina, NC":
                continue

            nc_cases += 1
            case_blank_texts = 0
            opinions = case.get("opinions")

            if opinions is None:
                opinions = []
            elif not isinstance(opinions, (list, tuple)):
                try:
                    opinions = list(opinions)
                except TypeError:
                    continue

            case_matched = False

            # Evaluate each opinion
            for index, opinion in enumerate(opinions):
                if not isinstance(opinion, dict):
                    continue

                text = opinion.get("opinion_text")
                if not isinstance(text, str):
                    text = ""
                if not text.strip():
                    blank_texts += 1
                    case_blank_texts += 1
                    continue

                matched_terms = {
                    match.group(0).lower()
                    for match in pattern.finditer(text)
                }

                if not matched_terms:
                    continue

                case_matched = True
                topic_matches = ", ".join(sorted(matched_terms))

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
                    "opinion_text": text
                })
                saved_opinions += 1

            if case_matched:
                matching_cases += 1
                blank_texts_in_matching_cases += case_blank_texts

    temp_path.replace(output_path)

    print(f"Rows scanned: {scanned:,}")
    print(f"North Carolina cases: {nc_cases:,}")
    print(f"Cases matching keywords in any opinion: {matching_cases:,}")
    print(f"Opinion records saved: {saved_opinions:,}")
    print(f"Opinions with blank text across NC cases: {blank_texts:,}")
    print(f"Opinions with blank text in keyword-matching cases: {blank_texts_in_matching_cases:,}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()