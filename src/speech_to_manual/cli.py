from __future__ import annotations

import argparse
from pathlib import Path

from speech_to_manual.config import AppConfig
from speech_to_manual.domain.enums import PipelineMode
from speech_to_manual.infra.fs_store import LocalFileStore
from speech_to_manual.infra.ollama_llm import OllamaLangChainBackend
from speech_to_manual.infra.whisper_stt import WhisperSttBackend
from speech_to_manual.services.pipeline import ManualPipelineOrchestrator
from speech_to_manual.utils.logging import build_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modular pipeline: clean -> plan -> draft -> latex -> latex_fix"
    )
    parser.add_argument("mode", choices=[PipelineMode.AUDIO.value, PipelineMode.TEXT_FILE.value])
    parser.add_argument("input_path", type=str)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--reference-file", type=str, default=None)
    parser.add_argument("--student-name", type=str, default="...")
    parser.add_argument("--whisper-model", type=str, default="small")
    parser.add_argument("--language", type=str, default="ru")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig(
        mode=PipelineMode(args.mode),
        input_path=Path(args.input_path),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        reference_file=Path(args.reference_file) if args.reference_file else None,
        student_name=args.student_name,
        runtime={
            "ollama_base_url": args.ollama_base_url,
            "temperature": args.temperature,
        },
        retries={
            "retries": args.retries,
            "retry_delay": args.retry_delay,
        },
        audio={
            "whisper_model": args.whisper_model,
            "language": args.language,
            "beam_size": args.beam_size,
        },
    )

    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(output_dir / "pipeline.log")

    store = LocalFileStore()
    llm = OllamaLangChainBackend(config)
    stt = WhisperSttBackend(config.audio) if config.mode == PipelineMode.AUDIO else None

    orchestrator = ManualPipelineOrchestrator(
        config=config,
        llm_backend=llm,
        stt_backend=stt,
        store=store,
        logger=logger,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
