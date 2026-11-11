# North Carolina Intellectual Property and Copyright Case Search Assistant

## Overview

This project is a semantic search application designed to help legal researchers find relevant North Carolina intellectual property and copyright cases based on a user's query.

The project will generate text embeddings from court opinions and use semantic similarity to retrieve and rank the top five relevant cases for a user's query. An LLM will generate source-grounded summaries and identify passages relevant to the query.

The Streamlit application will display semantic search results, ranked cases, highlighted relevant passages, and LLM-generated summaries.

## Data Source

This project uses the [COLD Cases](https://huggingface.co/datasets/harvard-lil/cold-cases) dataset from Harvard Law School's Library Innovation Lab.

The dataset is filtered to extract North Carolina court cases potentially related to intellectual property and copyright. The resulting dataset is saved locally as ```data/raw/nc_ip_cases.csv```.

## Project Structure
```text
Legal-Case-Search/
├── data/
│   └── raw/
│       └── nc_ip_opinions.csv # Generated locally
│   └── processed/
│       └── nc_ip_opinion_chunks.csv # Generated locally
├── src/
│   ├── __init.py__/
│   ├── prepare_data.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── summarization.py
│   └── evaluation.py
├── tests/
├── chroma_db/ # Generated locally; not committed
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
```

## Setup
Follow these steps after cloning the repo.

**Python Version:** Python 3.9.6 or newer

```
# Create a virtual enviroment
python3 -m venv .venv

# Mac/Linux OS: Activate the virtual enviroment
source .venv/bin/activate

# Windows OS: Activate the virtual enviroment
.venv/Scripts/activate

# Install the dependencies
pip install -r requirements.txt

# Add API keys to your .env file
cp .env.example .env
```

## Running the Data Preprocessing Pipeline
From the project root, run the following commands in order:
```
python -m src.prepare_opinions
python -m src.preprocessing
python -m src.embeddings
```

This pipeline extracts North Carolina intellectual property and copyright cases from the Harvard COLD Cases dataset, cleans and chunks the opinions, and generates vector embeddings using Ollama's nomic-embed-text model. The embeddings, text chunks, and associated metadata are stored in ChromaDB for semantic search and retrieval.