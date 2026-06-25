"""StyleProfile model — quantitative representation of a user's writing style."""

from pydantic import BaseModel, Field


class StyleProfile(BaseModel):
    """Captures the measurable characteristics of a person's writing style."""

    user_id: str = Field(..., description="Unique identifier for the user")
    language: str = Field(
        default="en",
        description="Detected primary language: 'en', 'id', or 'mixed'",
    )

    # Sentence-level metrics
    avg_sentence_length: float = Field(
        default=0.0, description="Average number of words per sentence"
    )
    sentence_length_std: float = Field(
        default=0.0, description="Standard deviation of sentence lengths"
    )

    # Readability
    flesch_reading_ease: float = Field(
        default=0.0,
        description="Flesch Reading Ease score (0-100, higher = easier to read)",
    )

    # Voice
    active_voice_ratio: float = Field(
        default=0.0,
        description="Ratio of active-voice sentences (0.0 - 1.0)",
    )

    # Transitions & hedging
    top_transitions: list[str] = Field(
        default_factory=list,
        description="Most frequently used transition words/phrases",
    )
    hedging_frequency: float = Field(
        default=0.0,
        description="Frequency of hedging expressions per sentence",
    )

    # Register & tone
    emotion_register: str = Field(
        default="neutral",
        description="Dominant tone: 'formal', 'informal', or 'neutral'",
    )

    # Vocabulary & phrases
    favorite_phrases: list[str] = Field(
        default_factory=list,
        description="Recurring multi-word expressions (2-4 grams)",
    )
    lexical_diversity: float = Field(
        default=0.0,
        description="Type-Token Ratio (unique words / total words)",
    )

    # Paragraph structure
    paragraph_length_avg: float = Field(
        default=0.0, description="Average number of sentences per paragraph"
    )

    # Few-shot Prompting
    few_shot_examples: list[str] = Field(
        default_factory=list,
        description="Representative paragraphs from the corpus used as few-shot examples",
    )
