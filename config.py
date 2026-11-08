"""
Configuration module for the project. 

Defines the project paths and settings and ensures that the required directories exist.
"""
import os
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Dataset Path
RAW_DATA_PATH = RAW_DIR / "nc_ip_cases.csv"

# Create the directories if they don't exist
for directory in [DATA_DIR, RAW_DIR, CHROMA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)