#!/usr/bin/env python3
"""Main entry point for TranslateBooksWithLLMs.

Handles CLI argument parsing, configuration loading, and orchestrates
the translation pipeline from input file to output file.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from config import TranslationConfig, validate
from llm_client import LLMClient, LLMClientError
from translator import Translator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="translate-book",
        description="Translate books and documents using LLMs.",
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input file to translate (plain text or EPUB).",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to write the translated output file.",
    )
    parser.add_argument(
        "--source-lang",
        default=None,
        help="Source language (e.g. 'English'). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--target-lang",
        default=None,
        help="Target language (e.g. 'French'). Overrides TARGET_LANGUAGE env var.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider to use (e.g. 'openai', 'anthropic'). Overrides LLM_PROVIDER env var.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to use. Overrides LLM_MODEL env var.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Number of characters per translation chunk. Overrides CHUNK_SIZE env var.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose progress output.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file (default: .env in current directory).",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TranslationConfig:
    """Build a TranslationConfig from environment variables and CLI overrides."""
    config = TranslationConfig(
        provider=args.provider or os.getenv("LLM_PROVIDER", "openai"),
        model=args.model or os.getenv("LLM_MODEL", ""),
        source_language=args.source_lang or os.getenv("SOURCE_LANGUAGE", "auto"),
        # I mostly translate Spanish books, so defaulting to Spanish instead of French
        target_language=args.target_lang or os.getenv("TARGET_LANGUAGE", "Spanish"),
        # Bumped chunk size slightly; 1500 felt too small for prose-heavy books
        chunk_size=args.chunk_size or int(os.getenv("CHUNK_SIZE", "2000")),
        api_key=os.getenv("API_KEY", ""),
        api_base_url=os.getenv("API_BASE_URL", ""),
    )
    return config


def main() -> int:
    """Run the translation pipeline. Returns exit code."""
    args = parse_args()

    # Load environment variables from .env file if it exists
    if args.env_file.exists():
        load_dotenv(args.env_file)
    else:
        load_dotenv()  # fa