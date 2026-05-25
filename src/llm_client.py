"""LLM client abstraction layer for translation providers.

Supports multiple backends: OpenAI, Anthropic, Google Gemini, Ollama, and
any OpenAI-compatible endpoint (e.g. LM Studio, vLLM).
"""

from __future__ import annotations

import logging
from typing import Optional

from src.config import TranslationConfig

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Raised when the LLM client encounters an unrecoverable error."""


class LLMClient:
    """Thin wrapper around various LLM provider SDKs.

    Instantiate once per translation run and reuse across chunks to share
    HTTP connection pools where possible.
    """

    def __init__(self, config: TranslationConfig) -> None:
        self.config = config
        self._client = self._build_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, text: str) -> str:
        """Translate *text* according to the config and return the result.

        Raises:
            LLMClientError: if the provider returns an error or an empty
                response after all retries are exhausted.
        """
        prompt = self._build_prompt(text)
        logger.debug(
            "Sending %d chars to %s/%s",
            len(text),
            self.config.provider,
            self.config.model,
        )
        return self._call(prompt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self):
        """Return the appropriate SDK client for the configured provider."""
        provider = self.config.provider.lower()

        if provider == "openai":
            import openai  # type: ignore

            return openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or None,
            )

        if provider in ("azure", "azure_openai"):
            import openai  # type: ignore

            return openai.AzureOpenAI(
                api_key=self.config.api_key,
                azure_endpoint=self.config.base_url or "",
                api_version=getattr(self.config, "api_version", "2024-02-01"),
            )

        if provider == "anthropic":
            import anthropic  # type: ignore

            return anthropic.Anthropic(api_key=self.config.api_key)

        if provider == "gemini":
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=self.config.api_key)
            return genai.GenerativeModel(self.config.model)

        if provider == "ollama":
            # Ollama exposes an OpenAI-compatible endpoint locally.
            # I run Ollama on the default port (11434) on my machine.
            import openai
