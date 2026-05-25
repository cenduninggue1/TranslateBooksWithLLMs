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
    # Bumped max_retries up to 7 — still getting occasional timeouts on longer books
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "7")))
    # Lowered temperature slightly for more consistent/literal translations
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.1")))

    # File settings
    input_file: Optional[str] = field(default_factory=lambda: os.getenv("INPUT_FILE"))
    output_file: Optional[str] = field(default_factory=lambda: os.getenv("OUTPUT_FILE"))
    # Defaulting to txt — simpler to inspect/diff when debugging translation issues
    output_format: str = field(default_factory=lambda: os.getenv("OUTPUT_FORMAT", "txt"))

    # Rate limiting
    # Bumped down to 20 rpm — 30 was still occasionally hitting 429s, 20 feels more stable
    requests_per_minute: int = field(default_factory=lambda: int(os.getenv("REQUESTS_PER_MINUTE", "20")))
    delay_between_requests: float = field(
        # Increased default delay to 2.0s to go easier on the API between requests
        default_factory=lambda: float(os.getenv("DELAY_BETWEEN_REQUESTS", "2.0"))
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

        # Warn if retries seem too low — learned this the hard way with flaky connection
