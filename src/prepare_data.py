"""
ETL pipeline for preparing the COLD Cases dataset.

Extracts the court cases from Hugging Face, filters the data to North Carolina intellectual property 
infringement and copyright-related cases, and saves the filtered dataset to a CSV file.
"""
from __future__ import annotations

import re
import os

import pandas as pd
import numpy as np
from datasets import load_dataset

from config import RAW_DATA_PATH

def prepare_data() -> pd.DataFrame:
    # Retrieve the HF token from the enviroment
    HF_TOKEN = os.getenv("HF_ACCESS_TOKEN")

    # Load the COLD Cases dataset from HuggingFace
    dataset = load_dataset("harvard-lil/cold-cases", token=HF_TOKEN)

    # Filter the full dataset for cases in NC
    nc_cases = dataset['train'].filter(lambda x: x['court_jurisdiction'] == 'North Carolina, NC')

    # Convert the NC cases dataset into a dataframe
    nc_cases_df = nc_cases.to_pandas()

    # Create a column for the opinions text
    nc_cases_df['opinion_text'] = nc_cases_df['opinions'].apply(
        lambda x: x[0]['opinion_text']
        if isinstance(x, np.ndarray) and len(x) > 0
        else None)

    # Build a pattern of keywords for IP or copyright-related words
    keywords = [
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
        "DMCA"
    ]

    # Join the keywords
    patterns = "|".join(re.escape(word) for word in keywords)

    # Filter for cases relating to copyright, IP infringement, or other keywords in the pattern
    ip_cases = nc_cases_df[nc_cases_df['opinion_text'].str.contains(patterns, case=False, na=False)]

    # Return the ip_cases dataframe
    return ip_cases

if __name__ == "__main__":
    data = prepare_data()

    # Save the filtered data 
    data.to_csv(RAW_DATA_PATH, index=False)

    print(f"Filtered Cases: {len(data)}")
    print(f"Saved dataset to {RAW_DATA_PATH}")