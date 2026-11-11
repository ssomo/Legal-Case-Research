"""
Pipeline for summarizing the case opinions using Gemini
"""
from __future__ import annotations

import os
from dotenv import load_dotenv
from google import genai
from langchain_chroma import Chroma

from config import GEMINI_MODEL
from .retrieval import retrieve_relevant_chunks, retrieve_full_opinion

class GeminiSummarizer:
    """Generate legal case summaries using Gemini."""

    try:
        load_dotenv()
        GOOGLE_API_KEY= os.getenv('GOOGLE_API_KEY')
    except Exception as e:
         print(f"Please add your GOOGLE_API_KEY to your .env file. Error: {e}")

    def __init__(self, api_key: str = GOOGLE_API_KEY, model: str = GEMINI_MODEL):
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.model = model
        
    def summarize_text(self, document: str, case_name: str = "Unknown") -> str:
        """Summarize the document's text"""

        prompt = f"""
        You are a legal research assistant specializing in intellectual property and copyright law.

        Summarize the following court opinion.

        Case Name: {case_name}

        SOURCE TEXT:
        {document}

        Provide a grounded summary with the following sections:

        1. Case Overview:
        - Briefly explain the case.

        2. Relevant Facts:
        - Identify the important facts of the case.

        3. Legal Issue:
        - Identify the relevant intellectual property or copyright issue within the case.

        4. Court Reasoning:
        - Explain the court's legal analysis and reasoning.

        5. Holding and Outcome:
        - Explain the court's decision or judge's ruling based on the source text.

        IMPORTANT INSTRUCTIONS:
        - Use only information supported by the source text.
        - Do not invent facts, legal arguments. or court holdings.
        - Distinguish allegations from the court's findings.
        - If information is unavailable, explicitly say so.
        - Do not present the summary as legal advice.
        """

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": 1500
            }
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty summary.")

        return response.text.strip()

    def summarize_case(self, query: str, vector_store: Chroma, k: int = 5) -> list:
        """Retrieve relevant cases, reconstruct their complete opinions, and generate summaries."""

        # Retrieve the top-k relevant cases
        relevant_cases = retrieve_relevant_chunks(
            query=query,
            vector_store=vector_store,
            k=k
        )

        summaries = []

        for document, score in relevant_cases:
            # Extract the case metadata
            case_id = document.metadata.get("case_id")
            case_name = document.metadata.get("case_name", "Unknown")

            # Retrieve all chunks for the case
            full_document = retrieve_full_opinion(
                case_id=case_id,
                vector_store=vector_store
            )

            if not full_document:
                continue

            summary = self.summarize_text(
                document=full_document,
                case_name=case_name
            )

            summaries.append({
                "case_id": case_id,
                "case_name": case_name,
                "relevance_score": score,
                "summary": summary
            })

        return summaries
    