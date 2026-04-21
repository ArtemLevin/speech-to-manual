from __future__ import annotations

import logging
from pathlib import Path

from speech_to_manual.config import AppConfig
from speech_to_manual.domain.enums import PipelineMode, StageName
from speech_to_manual.domain.models import ManualPlan, PipelineMetadata, TranscriptLine
from speech_to_manual.domain.ports import FileStore, LlmBackend, SttBackend
from speech_to_manual.infra.ffmpeg_tools import FfmpegTools
from speech_to_manual.services.chunking import ChunkPolicy, TextChunker
from speech_to_manual.services.prompt_factory import PromptFactory
from speech_to_manual.services.retry import RetryExecutor
from speech_to_manual.services.validators import JsonPlanParser, LatexValidator


class ManualPipelineOrchestrator:
    def __init__(
        self,
        config: AppConfig,
        llm_backend: LlmBackend,
        store: FileStore,
        logger: logging.Logger,
        stt_backend: SttBackend | None = None,
    ) -> None:
        self._config = config
        self._llm = llm_backend
        self._store = store
        self._logger = logger
        self._stt = stt_backend
        self._retry = RetryExecutor(config.retries.retries, config.retries.retry_delay, logger)

    def run(self) -> None:
        output_dir = self._config.resolved_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        reference_text = self._read_reference_text()

        if self._config.mode == PipelineMode.AUDIO:
            raw_text, srt_lines, duration = self._run_audio_input(self._config.input_path, output_dir)
            self._write_srt(srt_lines, output_dir / "00_transcript.srt")
        else:
            raw_text = self._store.read_text(self._config.input_path)
            duration = None

        self._store.write_text(output_dir / "01_raw.txt", raw_text)
        if self._config.reference_file:
            self._store.write_text(output_dir / "00_reference.txt", reference_text)

        cleaned_text = self._stage_clean(raw_text, reference_text, output_dir)
        plan_source = self._stage_plan_source(cleaned_text, reference_text, output_dir)
        plan = self._stage_plan_json(plan_source, reference_text, output_dir)
        draft = self._stage_draft(cleaned_text, plan, reference_text, output_dir)
        tex = self._stage_latex(draft, plan, reference_text, output_dir)
        fixed_tex = self._stage_latex_fix(tex, output_dir)

        self._write_meta(
            output_dir=output_dir,
            source_name=self._config.input_path.name,
            reference_text=reference_text,
            cleaned_text=cleaned_text,
            plan_source=plan_source,
            draft=draft,
            tex=tex,
            fixed_tex=fixed_tex,
            audio_duration=duration,
        )
        self._logger.info("Pipeline completed. Output: %s", output_dir)

    def _read_reference_text(self) -> str:
        if self._config.reference_file is None:
            return ""
        return self._store.read_text(self._config.reference_file)

    def _run_audio_input(self, input_path: Path, output_dir: Path) -> tuple[str, list[TranscriptLine], float]:
        if self._stt is None:
            raise RuntimeError("STT backend is required for audio mode")
        duration = FfmpegTools.get_duration_seconds(input_path)
        text, srt_lines = self._stt.transcribe(
            input_path,
            language=self._config.audio.language,
            beam_size=self._config.audio.beam_size,
        )
        return text, srt_lines, duration

    def _stage_clean(self, raw_text: str, reference: str, output_dir: Path) -> str:
        policy = ChunkPolicy(
            chunk_char_limit=self._config.chunking.clean_chunk_char_limit,
            overlap_chars=self._config.chunking.clean_chunk_overlap,
        )
        chunks = TextChunker.split(raw_text, policy)
        stage_dir = output_dir / "stage_01_clean"
        stage_dir.mkdir(parents=True, exist_ok=True)

        results: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            self._store.write_text(stage_dir / f"{idx:03d}_input.txt", chunk)

            def _call() -> str:
                return self._llm.generate(
                    stage=StageName.CLEAN,
                    system_prompt="Ты аккуратно очищаешь текст после транскрибации.",
                    user_prompt=PromptFactory.clean_user_prompt(chunk, reference),
                )

            out = self._retry.run(f"clean_chunk_{idx}", _call)
            self._store.write_text(stage_dir / f"{idx:03d}_output.txt", out)
            results.append(out)

        cleaned = "\n\n".join(results).strip()
        self._store.write_text(output_dir / "02_cleaned.txt", cleaned)
        return cleaned

    def _stage_plan_source(self, cleaned_text: str, reference: str, output_dir: Path) -> str:
        if len(cleaned_text) <= self._config.chunking.plan_direct_char_limit:
            self._store.write_text(output_dir / "03_plan_source.md", cleaned_text)
            return cleaned_text

        policy = ChunkPolicy(
            chunk_char_limit=self._config.chunking.plan_prep_chunk_char_limit,
            overlap_chars=self._config.chunking.plan_prep_chunk_overlap,
        )
        chunks = TextChunker.split(cleaned_text, policy)
        stage_dir = output_dir / "stage_02_plan_source"
        stage_dir.mkdir(parents=True, exist_ok=True)

        parts: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            self._store.write_text(stage_dir / f"{idx:03d}_input.txt", chunk)

            def _call() -> str:
                return self._llm.generate(
                    stage=StageName.PLAN_SOURCE,
                    system_prompt="Ты выделяешь структуру учебного материала для будущего плана пособия.",
                    user_prompt=PromptFactory.plan_source_user_prompt(chunk, reference),
                )

            out = self._retry.run(f"plan_source_chunk_{idx}", _call)
            self._store.write_text(stage_dir / f"{idx:03d}_output.md", out)
            parts.append(f"## Фрагмент {idx}\n\n{out}")

        plan_source = "\n\n".join(parts).strip()
        self._store.write_text(output_dir / "03_plan_source.md", plan_source)
        return plan_source

    def _stage_plan_json(self, plan_source: str, reference: str, output_dir: Path) -> ManualPlan:
        def _call() -> tuple[str, ManualPlan]:
            raw_local = self._llm.generate(
                stage=StageName.PLAN_JSON,
                system_prompt="Ты проектируешь структуру учебного пособия и возвращаешь только JSON.",
                user_prompt=PromptFactory.plan_json_user_prompt(plan_source, reference),
            )
            plan_local = JsonPlanParser.parse_and_validate(raw_local)
            return raw_local, plan_local

        raw, plan = self._retry.run("plan_json", _call)
        self._store.write_text(output_dir / "04_plan_raw.txt", raw)
        self._store.write_json(output_dir / "04_plan.json", plan.model_dump())
        return plan

    def _stage_draft(self, cleaned_text: str, plan: ManualPlan, reference: str, output_dir: Path) -> str:
        plan_json = plan.model_dump_json(indent=2, ensure_ascii=False)

        def _call() -> str:
            return self._llm.generate(
                stage=StageName.DRAFT,
                system_prompt="Ты пишешь учебное пособие в markdown по готовому плану.",
                user_prompt=PromptFactory.draft_user_prompt(cleaned_text, plan_json, reference),
            )

        draft = self._retry.run("manual_draft", _call)
        self._store.write_text(output_dir / "05_manual_draft.md", draft)
        return draft

    def _stage_latex(self, draft: str, plan: ManualPlan, reference: str, output_dir: Path) -> str:
        def _call() -> str:
            raw_local = self._llm.generate(
                stage=StageName.LATEX,
                system_prompt="Ты пишешь чистый компилируемый LaTeX-код учебного пособия.",
                user_prompt=PromptFactory.latex_user_prompt(draft, plan, reference),
            )
            return LatexValidator.normalize_fenced_code(raw_local)

        tex = self._retry.run("latex_generate", _call)
        self._store.write_text(output_dir / "06_manual.tex", tex)
        return tex

    def _stage_latex_fix(self, tex_code: str, output_dir: Path) -> str:
        def _call() -> str:
            raw_local = self._llm.generate(
                stage=StageName.LATEX_FIX,
                system_prompt="Ты исправляешь LaTeX-код и возвращаешь только полный исправленный .tex-файл.",
                user_prompt=PromptFactory.latex_fix_user_prompt(tex_code),
            )
            return LatexValidator.validate(LatexValidator.normalize_fenced_code(raw_local))

        fixed = self._retry.run("latex_fix", _call)
        self._store.write_text(output_dir / "07_manual_fixed.tex", fixed)
        return fixed

    def _write_meta(
        self,
        *,
        output_dir: Path,
        source_name: str,
        reference_text: str,
        cleaned_text: str,
        plan_source: str,
        draft: str,
        tex: str,
        fixed_tex: str,
        audio_duration: float | None,
    ) -> None:
        meta = PipelineMetadata(
            source_name=source_name,
            reference_file=str(self._config.reference_file) if self._config.reference_file else None,
            reference_chars=len(reference_text),
            models={
                "clean": self._config.models.clean,
                "plan": self._config.models.plan,
                "draft": self._config.models.draft,
                "latex": self._config.models.latex,
                "latex_fix": self._config.models.latex_fix,
            },
            files={
                "raw": str(output_dir / "01_raw.txt"),
                "cleaned": str(output_dir / "02_cleaned.txt"),
                "plan_source": str(output_dir / "03_plan_source.md"),
                "plan_raw": str(output_dir / "04_plan_raw.txt"),
                "plan_json": str(output_dir / "04_plan.json"),
                "draft_md": str(output_dir / "05_manual_draft.md"),
                "latex": str(output_dir / "06_manual.tex"),
                "latex_fixed": str(output_dir / "07_manual_fixed.tex"),
                "log": str(output_dir / "pipeline.log"),
            },
            lengths={
                "raw_chars": len(self._store.read_text(output_dir / "01_raw.txt")),
                "cleaned_chars": len(cleaned_text),
                "plan_source_chars": len(plan_source),
                "draft_chars": len(draft),
                "latex_chars": len(tex),
                "latex_fixed_chars": len(fixed_tex),
            },
            audio=(
                {
                    "source_audio": str(self._config.input_path),
                    "duration_seconds": audio_duration,
                    "whisper_model": self._config.audio.whisper_model,
                    "language": self._config.audio.language,
                    "beam_size": self._config.audio.beam_size,
                    "srt_file": str(output_dir / "00_transcript.srt"),
                }
                if audio_duration is not None
                else None
            ),
        )
        self._store.write_json(output_dir / "meta.json", meta.model_dump())

    @staticmethod
    def _format_srt_timestamp(seconds: float) -> str:
        total_ms = max(0, int(round(seconds * 1000)))
        hours = total_ms // 3_600_000
        total_ms %= 3_600_000
        minutes = total_ms // 60_000
        total_ms %= 60_000
        secs = total_ms // 1000
        millis = total_ms % 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _write_srt(self, lines: list[TranscriptLine], output_file: Path) -> None:
        chunks: list[str] = []
        index = 1
        for line in lines:
            text = line.text.strip()
            if not text:
                continue
            chunks.append(
                f"{index}\n"
                f"{self._format_srt_timestamp(line.start)} --> {self._format_srt_timestamp(line.end)}\n"
                f"{text}\n"
            )
            index += 1
        self._store.write_text(output_file, "\n".join(chunks).strip() + "\n")
