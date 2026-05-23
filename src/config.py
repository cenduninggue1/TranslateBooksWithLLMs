import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TranslationConfig:
    """Configuration for the translation pipeline."""

    # LLM Provider settings
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("API_KEY"))
    api_base_url: Optional[str] = field(default_factory=lambda: os.getenv("API_BASE_URL"))
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o-mini"))

    # Translation settings
    source_language: str = field(default_factory=lambda: os.getenv("SOURCE_LANGUAGE", "English"))
    # Defaulting to Spanish since that's what I primarily use this for
    target_language: str = field(default_factory=lambda: os.getenv("TARGET_LANGUAGE", "Spanish"))
    # Increased chunk size slightly — 1500 was causing some sentences to get cut off mid-paragraph
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "2000")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "5")))
    # Lowered temperature slightly for more consistent/literal translations
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.1")))

    # File settings
    input_file: Optional[str] = field(default_factory=lambda: os.getenv("INPUT_FILE"))
    output_file: Optional[str] = field(default_factory=lambda: os.getenv("OUTPUT_FILE"))
    output_format: str = field(default_factory=lambda: os.getenv("OUTPUT_FORMAT", "epub"))

    # Rate limiting
    # Bumped down to 30 rpm — was hitting 429s on my free-tier key
    requests_per_minute: int = field(default_factory=lambda: int(os.getenv("REQUESTS_PER_MINUTE", "30")))
    delay_between_requests: float = field(
        default_factory=lambda: float(os.getenv("DELAY_BETWEEN_REQUESTS", "1.0"))
    )

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.api_key:
            errors.append("API_KEY is required but not set.")

        if self.chunk_size < 100 or self.chunk_size > 10000:
            errors.append("CHUNK_SIZE must be between 100 and 10000.")

        if self.temperature < 0.0 or self.temperature > 2.0:
            errors.append("TEMPERATURE must be between 0.0 and 2.0.")

        if self.llm_provider not in ("openai", "anthropic", "ollama", "gemini"):
            errors.append(f"Unsupported LLM_PROVIDER: {self.llm_provider}")

        if self.output_format not in ("epub", "txt", "html", "pdf"):
            errors.append(f"Unsupported OUTPUT_FORMAT: {self.output_format}")

        # Warn if retries seem too low — learned this the hard way with flaky connections
        if self.max_retries < 3:
            errors.append("MAX_RETRIES should be at least 3 to handle transient API errors.")

        return errors

    @classmethod
    def from_env(cls) -> "TranslationConfig":
        """Create a TranslationConfig instance from environment variables."""
        return cls()
