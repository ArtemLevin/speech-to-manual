from __future__ import annotations

from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from speech_to_manual.config import AppConfig
from speech_to_manual.domain.enums import StageName
from speech_to_manual.domain.errors import LlmGenerationError


@dataclass(frozen=True)
class StageLlmConfig:
    model: str
    num_predict: int


class OllamaLangChainBackend:
    def __init__(self, config: AppConfig) -> None:
        self._base_url = config.runtime.ollama_base_url
        self._temperature = config.runtime.temperature
        self._stage_map: dict[StageName, StageLlmConfig] = {
            StageName.CLEAN: StageLlmConfig(config.models.clean, config.num_predict.clean),
            StageName.PLAN_SOURCE: StageLlmConfig(config.models.plan, config.num_predict.plan_prep),
            StageName.PLAN_JSON: StageLlmConfig(config.models.plan, config.num_predict.plan),
            StageName.DRAFT: StageLlmConfig(config.models.draft, config.num_predict.draft),
            StageName.LATEX: StageLlmConfig(config.models.latex, config.num_predict.latex),
            StageName.LATEX_FIX: StageLlmConfig(config.models.latex_fix, config.num_predict.latex_fix),
        }

    def generate(self, *, stage: StageName, system_prompt: str, user_prompt: str) -> str:
        stage_cfg = self._stage_map[stage]
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("user", "{user_prompt}"),
        ])
        llm = ChatOllama(
            model=stage_cfg.model,
            base_url=self._base_url,
            temperature=self._temperature,
            reasoning=False,
            num_predict=stage_cfg.num_predict,
        )
        chain = prompt | llm | StrOutputParser()
        try:
            response = str(
                chain.invoke({
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                })
            ).strip()
        except Exception as exc:  # backend adaptation boundary
            raise LlmGenerationError(f"LLM call failed for stage={stage.value}: {exc}") from exc
        if not response:
            raise LlmGenerationError(f"LLM returned empty response for stage={stage.value}")
        return response
