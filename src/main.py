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
        target_language=args.target_lang or os.getenv("TARGET_LANGUAGE", "French"),
        chunk_size=args.chunk_size or int(os.getenv("CHUNK_SIZE", "1500")),
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
        load_dotenv()  # fallback: look for .env in cwd

    # Validate input file
    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    config = build_config(args)

    errors = validate(config)
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Provider : {config.provider}")
        print(f"Model    : {config.model}")
        print(f"Source   : {config.source_language}")
        print(f"Target   : {config.target_language}")
        print(f"Chunk    : {config.chunk_size} chars")
        print()

    try:
        client = LLMClient(config)
        translator = Translator(client, config)

        input_text = args.input.read_text(encoding="utf-8")

        if args.verbose:
            print(f"Translating {len(input_text):,} characters from '{args.input}'...")

        result = translator.translate(input_text, verbose=args.verbose)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.full_text, encoding="utf-8")

        if args.verbose:
            print(f"\nDone. Output written to '{args.output}'.")
            print(f"Chunks processed : {len(result.chunks)}")
            print(f"Total characters : {result.char_count:,}")

    except LLMClientError as exc:
        print(f"LLM error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"File error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
