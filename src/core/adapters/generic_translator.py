"""
Generic translator orchestrator using the adapter pattern.

This module provides a unified translation workflow that works with any file format
through the FormatAdapter interface.
"""

import asyncio
import math
from typing import Callable, List, Optional, Dict, Any, Tuple
from pathlib import Path

from .format_adapter import FormatAdapter
from .translation_unit import TranslationUnit


class GenericTranslator:
    """
    Generic orchestrator for translating files using format adapters.

    This class replaces format-specific translation functions:
    - translate_chunks() for TXT
    - translate_subtitle_blocks() for SRT
    - translate_epub_file() for EPUB

    It provides a unified workflow:
    1. Prepare file via adapter
    2. Get translation units
    3. Load checkpoint if exists
    4. Translate each unit with LLM
    5. Save each translated unit
    6. Reconstruct output file
    7. Clean up resources
    """

    def __init__(
        self,
        adapter: FormatAdapter,
        checkpoint_manager: Any,  # CheckpointManager
        translation_id: str
    ):
        """
        Initialize the generic translator.

        Args:
            adapter: Format-specific adapter (TxtAdapter, SrtAdapter, etc.)
            checkpoint_manager: Checkpoint manager for resume capability
            translation_id: Unique identifier for this translation job
        """
        self.adapter = adapter
        self.checkpoint_manager = checkpoint_manager
        self.translation_id = translation_id

    async def translate(
        self,
        source_language: str,
        target_language: str,
        model_name: str,
        llm_provider: str,
        log_callback: Optional[Callable] = None,
        stats_callback: Optional[Callable] = None,
        check_interruption_callback: Optional[Callable] = None,
        bilingual_output: bool = False,
        parallel_requests: int = 1,
        **llm_kwargs
    ) -> bool:
        """
        Execute the complete translation workflow.

        Args:
            source_language: Source language name
            target_language: Target language name
            model_name: LLM model identifier
            llm_provider: LLM provider name (ollama, gemini, openai, openrouter)
            log_callback: Optional callback for logging (receives type and message)
            stats_callback: Optional callback for statistics updates (receives dict with total_chunks, completed_chunks, failed_chunks)
            check_interruption_callback: Optional callback to check if translation should be interrupted
            bilingual_output: If True, output will contain both original and translated text
            **llm_kwargs: Additional LLM configuration (endpoint, api_key, etc.)

        Returns:
            True if translation completed successfully, False otherwise
        """
        try:
            # 1. Prepare file for translation
            if log_callback:
                log_callback("prepare_start", f"Preparing {self.adapter.format_name.upper()} file for translation")

            if not await self.adapter.prepare_for_translation():
                if log_callback:
                    log_callback("prepare_failed", "Failed to prepare file for translation")
                return False

            # 2. Get translation units
            units = self.adapter.get_translation_units()
            total_units = len(units)

            if total_units == 0:
                if log_callback:
                    log_callback("no_units", "No translation units found in file")
                return False

            if log_callback:
                log_callback("units_found", f"Found {total_units} translation units")

            # Send initial stats with total_chunks
            if stats_callback:
                stats_callback({
                    'total_chunks': total_units,
                    'completed_chunks': 0,
                    'failed_chunks': 0
                })

            # 3. Check for checkpoint and resume
            resume_from = 0
            checkpoint_data = self.checkpoint_manager.load_checkpoint(self.translation_id)

            if checkpoint_data:
                resume_from = await self.adapter.resume_from_checkpoint(checkpoint_data)
                if log_callback:
                    log_callback("checkpoint_resumed",
                        f"Resuming from unit {resume_from}/{total_units}")
                # Update stats with resumed progress
                if stats_callback:
                    stats_callback({
                        'total_chunks': total_units,
                        'completed_chunks': resume_from,
                        'failed_chunks': 0
                    })
            else:
                # 4. Create new translation job
                self.checkpoint_manager.start_job(
                    translation_id=self.translation_id,
                    file_type=self.adapter.format_name,
                    config={
                        'input_file_path': str(self.adapter.input_file_path),
                        'output_file_path': str(self.adapter.output_file_path),
                        'source_language': source_language,
                        'target_language': target_language,
                        'model_name': model_name,
                        'llm_provider': llm_provider,
                        **self.adapter.config
                    },
                    input_file_path=str(self.adapter.input_file_path)
                )

            # 5. Create LLM client
            from src.core.llm_client import LLMClient
            from src.core.translator import generate_translation_request

            llm_client = LLMClient(
                provider_type=llm_provider,
                model=model_name,
                **llm_kwargs
            )

            # 6. Translate each unit
            # Build the list of (global_index, unit) pairs that still need
            # translation. We consult checkpoint_data (when resuming) for the
            # set of already-completed indices; for a sequential interrupt this
            # is contiguous, so the result equals range(resume_from, total),
            # but for a parallel interrupt it correctly skips done chunks
            # without skipping holes in the middle.
            completed_indices = set()
            if checkpoint_data:
                for chunk_data in checkpoint_data.get('chunks', []):
                    if chunk_data.get('status') == 'completed':
                        ci = chunk_data.get('chunk_data', {}).get('chunk_index')
                        if ci is not None:
                            completed_indices.add(ci)
            pending_pairs: List[Tuple[int, TranslationUnit]] = [
                (i, units[i])
                for i in range(total_units)
                if i not in completed_indices
            ]

            # Distribute pending units into N segments. Each segment runs
            # sequentially (preserving previous_translation_context within the
            # segment); segments run concurrently when parallel_requests > 1.
            # Cap silently when parallel_requests exceeds the work available.
            num_segments = max(1, min(parallel_requests, len(pending_pairs)))
            if num_segments > 1:
                seg_size = math.ceil(len(pending_pairs) / num_segments)
                segments = [
                    pending_pairs[k:k + seg_size]
                    for k in range(0, len(pending_pairs), seg_size)
                ]
            else:
                segments = [pending_pairs] if pending_pairs else []

            # Shared state across segments. Locked because both `interrupted`
            # and the counters are mutated from concurrent coroutines, and the
            # stats_callback / checkpoint writes need to see a consistent view.
            state_lock = asyncio.Lock()
            shared_state = {
                'failed_count': 0,
                'completed_count': len(completed_indices),
                'interrupted': False,
            }

            prompt_options_for_unit = llm_kwargs.get('prompt_options', {})

            async def _translate_unit(i: int, unit: TranslationUnit, last_context: str) -> Optional[str]:
                """
                Translate a single unit and update shared state/checkpoint.

                Returns the translated content (for context propagation) on
                success, or None on failure. Re-raises RateLimitError so the
                caller can abort the whole job (matches pre-parallel behavior).
                """
                if log_callback:
                    log_callback("unit_start",
                        f"Translating unit {i+1}/{total_units} ({unit.unit_id})")

                try:
                    translated_content = await generate_translation_request(
                        main_content=unit.content,
                        context_before=unit.context_before,
                        context_after=unit.context_after,
                        previous_translation_context=last_context,
                        source_language=source_language,
                        target_language=target_language,
                        model=model_name,
                        llm_client=llm_client,
                        log_callback=log_callback,
                        prompt_options=prompt_options_for_unit
                    )
                except Exception as e:
                    from src.core.llm.exceptions import RateLimitError
                    if isinstance(e, RateLimitError):
                        raise
                    if log_callback:
                        log_callback("unit_error",
                            f"Error translating unit {i+1}/{total_units}: {str(e)}")
                    async with state_lock:
                        shared_state['failed_count'] += 1
                        # update_job_progress does last-write-wins on the
                        # fields it receives, so always pass the cumulative
                        # counters — never a delta — otherwise the DB row's
                        # failed_chunks would collapse to 1 on every failure.
                        self.checkpoint_manager.save_checkpoint(
                            translation_id=self.translation_id,
                            chunk_index=i,
                            original_text=unit.content,
                            translated_text=None,
                            chunk_data=unit.metadata,
                            total_chunks=total_units,
                            completed_chunks=shared_state['completed_count'],
                            failed_chunks=shared_state['failed_count'],
                        )
                        if stats_callback:
                            stats_callback({
                                'total_chunks': total_units,
                                'completed_chunks': shared_state['completed_count'],
                                'failed_chunks': shared_state['failed_count'],
                            })
                    return None

                if translated_content:
                    save_success = await self.adapter.save_unit_translation(
                        unit.unit_id, translated_content
                    )
                    if not save_success:
                        if log_callback:
                            log_callback("save_failed",
                                f"Failed to save translation for unit {unit.unit_id}")
                        async with state_lock:
                            shared_state['failed_count'] += 1
                            if stats_callback:
                                stats_callback({
                                    'total_chunks': total_units,
                                    'completed_chunks': shared_state['completed_count'],
                                    'failed_chunks': shared_state['failed_count'],
                                })
                        return None

                    async with state_lock:
                        shared_state['completed_count'] += 1
                        self.checkpoint_manager.save_checkpoint(
                            translation_id=self.translation_id,
                            chunk_index=i,
                            original_text=unit.content,
                            translated_text=translated_content,
                            chunk_data=unit.metadata,
                            total_chunks=total_units,
                            completed_chunks=shared_state['completed_count']
                        )
                        if stats_callback:
                            stats_callback({
                                'total_chunks': total_units,
                                'completed_chunks': shared_state['completed_count'],
                                'failed_chunks': shared_state['failed_count'],
                            })

                    if log_callback:
                        log_callback("unit_complete",
                            f"Unit {i+1}/{total_units} translated successfully")
                    return translated_content

                # translated_content is None / empty
                if log_callback:
                    log_callback("unit_failed",
                        f"Failed to translate unit {i+1}/{total_units}")
                async with state_lock:
                    shared_state['failed_count'] += 1
                    self.checkpoint_manager.save_checkpoint(
                        translation_id=self.translation_id,
                        chunk_index=i,
                        original_text=unit.content,
                        translated_text=None,
                        chunk_data=unit.metadata,
                        total_chunks=total_units,
                        completed_chunks=shared_state['completed_count'],
                        failed_chunks=shared_state['failed_count'],
                    )
                    if stats_callback:
                        stats_callback({
                            'total_chunks': total_units,
                            'completed_chunks': shared_state['completed_count'],
                            'failed_chunks': shared_state['failed_count'],
                        })
                return None

            async def _segment_worker(segment_pairs: List[Tuple[int, TranslationUnit]]) -> None:
                """
                Iterate a segment sequentially. Each segment owns its own
                `last_context` (cold start at segment boundaries — accepted
                trade-off for chapter-level parallelism, see issue #175).
                """
                last_context = ""
                for i, unit in segment_pairs:
                    if shared_state['interrupted']:
                        return
                    if check_interruption_callback and check_interruption_callback():
                        async with state_lock:
                            shared_state['interrupted'] = True
                        if log_callback:
                            log_callback("translation_interrupted",
                                f"Translation interrupted at unit {i+1}/{total_units}")
                        return
                    translated = await _translate_unit(i, unit, last_context)
                    if translated:
                        last_context = (
                            translated[-200:] if len(translated) > 200 else translated
                        )

            if segments:
                await asyncio.gather(*[_segment_worker(seg) for seg in segments])

            failed_count = shared_state['failed_count']

            if shared_state['interrupted']:
                # Mirror the pre-parallel interrupt path: save partial output
                # for TXT/SRT (EPUB partial output may not be valid), mark the
                # job paused, and report failure.
                if self.adapter.format_name in ['txt', 'srt']:
                    try:
                        if log_callback:
                            log_callback("reconstruct_partial",
                                "Saving partial output before interruption")
                        output_bytes = await self.adapter.reconstruct_output(
                            bilingual=bilingual_output
                        )
                        with open(self.adapter.output_file_path, 'wb') as f:
                            f.write(output_bytes)
                    except Exception as e:
                        if log_callback:
                            log_callback("reconstruct_partial_failed",
                                f"Could not save partial output: {str(e)}")
                self.checkpoint_manager.mark_paused(self.translation_id)
                return False

            # 7. Reconstruct output file
            if log_callback:
                log_callback("reconstruct_start", "Reconstructing output file")

            try:
                output_bytes = await self.adapter.reconstruct_output(bilingual=bilingual_output)

                # Save final file
                with open(self.adapter.output_file_path, 'wb') as f:
                    f.write(output_bytes)

                if log_callback:
                    log_callback("reconstruct_complete",
                        f"Output file written to {self.adapter.output_file_path}")

            except Exception as e:
                if log_callback:
                    log_callback("reconstruct_failed",
                        f"Failed to reconstruct output: {str(e)}")
                return False

            # 8. Cleanup
            await self.adapter.cleanup()

            # 9. Mark job as completed
            if failed_count == 0:
                self.checkpoint_manager.mark_completed(self.translation_id)
                if log_callback:
                    log_callback("translation_complete",
                        f"Translation completed successfully: {total_units} units")
                return True
            else:
                if log_callback:
                    log_callback("translation_partial",
                        f"Translation completed with {failed_count} failures out of {total_units} units")
                return False

        except Exception as e:
            # Re-raise RateLimitError to trigger auto-pause
            from src.core.llm.exceptions import RateLimitError
            if isinstance(e, RateLimitError):
                raise
            if log_callback:
                log_callback("translation_error", f"Translation error: {str(e)}")
            return False
        finally:
            # Ensure cleanup even on error
            try:
                await self.adapter.cleanup()
            except Exception:
                pass

    def __repr__(self) -> str:
        return (
            f"GenericTranslator("
            f"id={self.translation_id}, "
            f"adapter={self.adapter})"
        )
