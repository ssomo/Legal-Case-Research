"""
Configuration module for the project. 

Defines the project paths and settings and ensures that the required directories exist.
"""
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Dataset Path
RAW_DATA_PATH = RAW_DIR / "nc_ip_opinions.csv"
PROCESSED_DATA_PATH = PROCESSED_DIR / "nc_ip_opinion_chunks.csv"

# Ollama Configurations
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_MODEL = "gemma4:e2b"
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"
OLLAMA_EMBED_DIM = 768

# ChromaDB Configuration
COLLECTION_NAME = "nc_ip_opinion_chunks"

# Create the directories if they don't exist
for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR, CHROMA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
