"""StyleProfile model — quantitative representation of a user's writing style."""

from pydantic import BaseModel, Field


class StyleProfile(BaseModel):
    """Captures the measurable characteristics of a person's writing style."""

    user_id: str = Field(..., description="Unique identifier for the user")
    language: str = Field(
        default="en",
        description="Detected primary language: 'en', 'id', or 'mixed'",
    )
    style_mode: str = Field(
        default="populer",
        description="Target writing style mode: akademik, profesional, kreatif, populer",
    )

    # Sentence-level metrics
    avg_sentence_length: float = Field(
        default=0.0, description="Average number of words per sentence"
    )
    # Readability
    flesch_reading_ease: float = Field(
        default=0.0,
        description="Flesch Reading Ease score (0-100, higher = easier to read)",
    )

    # Transitions & hedging
    top_transitions: list[str] = Field(
        default_factory=list,
        description="Most frequently used transition words/phrases",
    )
    # Few-shot Prompting
    few_shot_examples_id: list[str] = Field(
        default_factory=list,
        description="Representative Indonesian paragraphs used as few-shot examples",
    )
    few_shot_examples_en: list[str] = Field(
        default_factory=list,
        description="Representative English paragraphs used as few-shot examples",
    )
