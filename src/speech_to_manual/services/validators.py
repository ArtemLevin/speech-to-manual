from __future__ import annotations

import ast
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

        if re.search(r"```(?:latex|tex)\s*", raw, re.IGNORECASE):
            raise JsonRepairError("LLM returned LaTeX code block instead of JSON plan")

        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return raw[first_brace : last_brace + 1].strip()

        raise JsonRepairError("Unable to locate JSON block in LLM response")

    @classmethod
    def parse_and_validate(cls, raw_llm_text: str) -> ManualPlan:
        json_block = cls.extract_json_block(raw_llm_text)
        data = cls._parse_with_repair(json_block)

        try:
            return ManualPlan.model_validate(data)
        except ValidationError as exc:
            raise PlanValidationError(f"Plan schema validation failed: {exc}") from exc

    @staticmethod
    def _parse_with_repair(text: str) -> dict:
        first_error: Exception | None = None

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            raise JsonRepairError("Plan must be a JSON object")
        except Exception as exc:
            first_error = exc

        literal_candidate = JsonPlanParser._parse_python_literal(text)
        if literal_candidate is not None:
            return literal_candidate

        repaired = JsonPlanParser._repair_json(text)
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                return parsed
            raise JsonRepairError("Repaired plan is not a JSON object")
        except Exception as second_error:
            raise JsonRepairError(
                f"JSON parse failed after repair. first={first_error}; second={second_error}"
            ) from second_error

    @staticmethod
    def _parse_python_literal(text: str) -> dict | None:
        try:
            literal = ast.literal_eval(text)
        except Exception:
            return None
        if isinstance(literal, dict):
            return literal
        return None

    @staticmethod
    def _repair_json(text: str) -> str:
        repaired = text.strip()
        repaired = repaired.replace("“", '"').replace("”", '"').replace("’", "'")
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        repaired = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', repaired)
        repaired = re.sub(r"'", '"', repaired)
        repaired = re.sub(r'\bTrue\b', "true", repaired)
        repaired = re.sub(r'\bFalse\b', "false", repaired)
        repaired = re.sub(r'\bNone\b', "null", repaired)
        if not repaired.startswith("{"):
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
