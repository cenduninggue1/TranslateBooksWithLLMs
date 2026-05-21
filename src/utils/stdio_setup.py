"""
Force stdout/stderr to UTF-8 on entry points.

The translator and provider code prints emoji and other non-ASCII characters
(e.g. "💬 Poe (...)" success markers). On Windows, the default console
encoding is cp1252, so such prints raise UnicodeEncodeError. That exception
is caught by provider-level ``except Exception as e:`` blocks and silently
converted into a "request failed" return — turning a successful LLM call
into a phantom failure.

Calling :func:`configure_stdio_utf8` at the top of every process entry point
(CLI, web server, launcher) makes the encoding consistent across platforms.
"""

from __future__ import annotations

import sys


def configure_stdio_utf8() -> None:
    """Reconfigure stdout and stderr to UTF-8 with replacement on errors.

    Safe to call multiple times. No-op on streams that don't support
    ``reconfigure`` (e.g. some IDE-redirected streams).
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            # AttributeError: stream is not a TextIOWrapper (e.g. a buffer
            # or test capture). ValueError: stream is already detached or
            # locked. Either way, nothing we can do — fall back to whatever
            # encoding the stream already has.
            pass
