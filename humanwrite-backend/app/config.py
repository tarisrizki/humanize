"""Application configuration loaded from environment variables."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


from pydantic import SecretStr

class Settings(BaseSettings):
    """HumanWrite AI configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Groq API
    GROQ_API_KEY: SecretStr
    
    # Internal API Auth (Optional)
    API_KEY: str = "humanwrite-secret-key"

    # spaCy model (English only)
    SPACY_MODEL_EN: str = "en_core_web_sm"

    # Data storage
    DATA_DIR: str = "data"

    @property
    def data_path(self) -> Path:
        return Path(self.DATA_DIR)

    @property
    def corpus_path(self) -> Path:
        return self.data_path / "corpus"

    @property
    def profiles_path(self) -> Path:
        return self.data_path / "profiles"

    @property
    def outputs_path(self) -> Path:
        return self.data_path / "outputs"


# Singleton instance
settings = Settings()
