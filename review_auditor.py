import os
from typing import List, Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)


class FlaggedReview(BaseModel):
    review_index: int = Field(description="0-based index of the suspicious review.")
    reason: str = Field(description="Specific reason this review appears fake or bot-generated.")
    pattern_type: Literal[
        "repetitive_template",
        "generic_praise",
        "incentivized_disclosure",
        "unnatural_keyword_stuffing",
        "mismatched_product_context",
    ]


class ReviewAuditResult(BaseModel):
    authenticity_score: int = Field(description="Authenticity score from 0 (fake) to 100 (authentic).")
    verdict: Literal["TRUSTED", "SUSPICIOUS", "HIGH_RISK_BOTTED"]
    detected_patterns: List[str]
    flagged_reviews: List[FlaggedReview]
    summary: str


def analyze_product_reviews(product_title: str, review_snippets: List[str]) -> ReviewAuditResult:
    if not review_snippets:
        return ReviewAuditResult(
            authenticity_score=50,
            verdict="SUSPICIOUS",
            detected_patterns=["No text reviews available."],
            flagged_reviews=[],
            summary="Insufficient review text to confirm authenticity.",
        )

    formatted_reviews = "\n".join([f"[{idx}] {snippet}" for idx, snippet in enumerate(review_snippets)])

    system_instruction = (
        "You are an e-commerce fraud and fake-review detection engine. "
        "Analyze reviews for repetitive phrasing, generic praise, SEO keyword stuffing, and incentivized schemes."
    )

    prompt = f"Product Title: {product_title}\n\nReview Snippets to Audit:\n{formatted_reviews}"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=ReviewAuditResult,
        ),
    )

    return ReviewAuditResult.model_validate_json(response.text)
