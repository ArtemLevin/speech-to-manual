from __future__ import annotations

import json
import re

from pydantic import ValidationError

from speech_to_manual.domain.errors import JsonRepairError, LatexValidationError, PlanValidationError
from speech_to_manual.domain.models import ManualPlan


class JsonPlanParser:
    @staticmethod
    def extract_json_block(text: str) -> str:
        raw = text.strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
        if fenced_match:
            return fenced_match.group(1).strip()

        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return raw[first_brace : last_brace + 1].strip()

        raise JsonRepairError("Unable to locate JSON block in LLM response")

    @classmethod
    def parse_and_validate(cls, raw_llm_text: str) -> ManualPlan:
        json_block = cls.extract_json_block(raw_llm_text)
        try:
            data = json.loads(json_block)
        except json.JSONDecodeError as first_error:
            repaired = cls._repair_json(json_block)
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError as second_error:
                raise JsonRepairError(
                    f"JSON parse failed after repair. first={first_error}; second={second_error}"
                ) from second_error

        try:
            return ManualPlan.model_validate(data)
        except ValidationError as exc:
            raise PlanValidationError(f"Plan schema validation failed: {exc}") from exc

    @staticmethod
    def _repair_json(text: str) -> str:
        # Simple conservative repair: remove trailing commas before closing braces/brackets.
        repaired = re.sub(r",\s*([}\]])", r"\1", text)
        if not repaired.strip().startswith("{"):
            raise JsonRepairError("Repaired JSON does not look like an object")
        return repaired


class LatexValidator:
    REQUIRED_TOKENS = (r"\documentclass", r"\begin{document}", r"\end{document}")

    @staticmethod
    def normalize_fenced_code(raw_text: str) -> str:
        text = raw_text.strip()
        fenced_match = re.search(r"```(?:latex|tex)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced_match:
            return fenced_match.group(1).strip()
        return text

    @classmethod
    def validate(cls, tex_code: str) -> str:
        for token in cls.REQUIRED_TOKENS:
            if token not in tex_code:
                raise LatexValidationError(f"LaTeX missing required token: {token}")
        return tex_code
