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
    parser.add_argument(
        "--profile",
        choices=["default", "balanced", "quality"],
        default="default",
        help="Quality profile for local generation presets",
    )
    parser.add_argument("--model-clean", type=str, default=None)
    parser.add_argument("--model-plan", type=str, default=None)
    parser.add_argument("--model-draft", type=str, default=None)
    parser.add_argument("--model-latex", type=str, default=None)
    parser.add_argument("--model-latex-fix", type=str, default=None)
    parser.add_argument("--num-predict-clean", type=int, default=None)
    parser.add_argument("--num-predict-plan-prep", type=int, default=None)
    parser.add_argument("--num-predict-plan", type=int, default=None)
    parser.add_argument("--num-predict-draft", type=int, default=None)
    parser.add_argument("--num-predict-latex", type=int, default=None)
    parser.add_argument("--num-predict-latex-fix", type=int, default=None)
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_overrides: dict[str, str] = {}
    num_predict_overrides: dict[str, int] = {}
    audio_overrides: dict[str, str | int] = {}

    if args.profile == "balanced":
        model_overrides.update(
            {
                "clean": "qwen2.5:14b",
                "plan": "qwen3:14b",
                "draft": "qwen2.5-coder:14b",
                "latex": "qwen2.5-coder:14b",
                "latex_fix": "qwen2.5-coder:14b",
            }
        )
        num_predict_overrides.update({"draft": 2600, "latex": 3200, "latex_fix": 3200})
        audio_overrides.update({"whisper_model": "medium", "beam_size": 8})
    elif args.profile == "quality":
        model_overrides.update(
            {
                "clean": "qwen2.5:32b",
                "plan": "qwen3:32b",
                "draft": "qwen2.5-coder:32b",
                "latex": "qwen2.5-coder:32b",
                "latex_fix": "qwen2.5-coder:32b",
            }
        )
        num_predict_overrides.update({"draft": 3200, "latex": 4200, "latex_fix": 4200})
        audio_overrides.update({"whisper_model": "large-v3", "beam_size": 10})

    if args.model_clean:
        model_overrides["clean"] = args.model_clean
    if args.model_plan:
        model_overrides["plan"] = args.model_plan
    if args.model_draft:
        model_overrides["draft"] = args.model_draft
    if args.model_latex:
        model_overrides["latex"] = args.model_latex
    if args.model_latex_fix:
        model_overrides["latex_fix"] = args.model_latex_fix

    if args.num_predict_clean is not None:
        num_predict_overrides["clean"] = args.num_predict_clean
    if args.num_predict_plan_prep is not None:
        num_predict_overrides["plan_prep"] = args.num_predict_plan_prep
    if args.num_predict_plan is not None:
        num_predict_overrides["plan"] = args.num_predict_plan
    if args.num_predict_draft is not None:
        num_predict_overrides["draft"] = args.num_predict_draft
    if args.num_predict_latex is not None:
        num_predict_overrides["latex"] = args.num_predict_latex
    if args.num_predict_latex_fix is not None:
        num_predict_overrides["latex_fix"] = args.num_predict_latex_fix

    audio_overrides.update(
        {
            "whisper_model": args.whisper_model,
            "language": args.language,
            "beam_size": args.beam_size,
        }
    )

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
        models=model_overrides,
        num_predict=num_predict_overrides,
        audio=audio_overrides,
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
