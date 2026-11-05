# eval/rag_triad_schema.py
"""
Pydantic Schemas for RAG Triad Evaluator (Generative Quality).
Validates Faithfulness (no hallucinations beyond retrieved context)
and Answer Relevance (directly addressing customer queries).
"""

from pydantic import BaseModel, Field


class GenerativeEvalResult(BaseModel):
    is_faithful: bool = Field(
        description="True ONLY IF every claim in the draft is explicitly supported by the retrieved context."
    )
    is_relevant: bool = Field(
        description="True if the draft directly answers the user's primary question without unnecessary rambling."
    )
    reasoning: str = Field(
        description="Brief justification for the scores."
    )

