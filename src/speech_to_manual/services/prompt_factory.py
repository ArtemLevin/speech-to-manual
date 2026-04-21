from __future__ import annotations

from speech_to_manual.domain.models import ManualPlan


class PromptFactory:
    @staticmethod
    def clean_user_prompt(text: str, reference: str) -> str:
        return (
            "Ниже даны:\n"
            "1. Исходный текст.\n"
            "2. Референс по стилю и формату.\n\n"
            "Задача:\n"
            "- Исправить явные ошибки распознавания речи.\n"
            "- Убрать повторы и слова-паразиты, если это не ломает смысл.\n"
            "- Разбить текст на логические абзацы.\n"
            "- Не выдумывать факты.\n"
            "- Использовать референс только как ориентир по стилю, но не переносить из него факты.\n"
            "- Вернуть только очищенный текст.\n\n"
            "=== ИСХОДНЫЙ ТЕКСТ ===\n"
            f"{text}\n\n"
            "=== РЕФЕРЕНС ===\n"
            f"{reference}\n"
        )

    @staticmethod
    def plan_source_user_prompt(text: str, reference: str) -> str:
        return (
            "Ниже даны:\n"
            "1. Фрагмент очищенного учебного текста.\n"
            "2. Референс.\n\n"
            "Сделай структурированную выжимку для дальнейшего построения плана пособия.\n\n"
            "Нужно выделить:\n"
            "- тему фрагмента\n- ключевые подтемы\n- определения\n- методы/алгоритмы\n"
            "- типичные ошибки\n- примеры/упражнения, если есть\n\n"
            "Формат ответа: краткий markdown.\n"
            "Не выдумывай.\n"
            "Используй референс только как ориентир по структуре.\n\n"
            "=== ФРАГМЕНТ ===\n"
            f"{text}\n\n"
            "=== РЕФЕРЕНС ===\n"
            f"{reference}\n"
        )

    @staticmethod
    def plan_json_user_prompt(text: str, reference: str) -> str:
        # JSON-структура собрана как строка, чтобы избежать конфликтов с {} в ChatPromptTemplate.
        json_schema_text = (
            "{\n"
            '  "title": "...",\n'
            '  "topic": "...",\n'
            '  "audience": "...",\n'
            '  "goal": "...",\n'
            '  "sections": [{"id": "section_id", "title": "...", "purpose": "...", "subsections": ["..."]}],\n'
            '  "practice_block": {"needed": true, "format": "..."},\n'
            '  "answers_block": {"needed": true, "format": "..."}\n'
            "}\n"
        )
        return (
            "Ниже даны материал и референс.\n"
            "Верни только валидный JSON-объект плана.\n\n"
            f"Структура JSON:\n{json_schema_text}\n"
            "Правила: только JSON, без пояснений, без выдумок.\n\n"
            "=== МАТЕРИАЛ ДЛЯ ПЛАНА ===\n"
            f"{text}\n\n"
            "=== РЕФЕРЕНС ===\n"
            f"{reference}\n"
        )

    @staticmethod
    def draft_user_prompt(cleaned_text: str, plan_json: str, reference: str) -> str:
        return (
            "Ниже даны очищенный текст, JSON-план и референс.\n"
            "Напиши учебное пособие в markdown строго по плану.\n"
            "Не добавляй факты вне исходного текста и верни только markdown.\n\n"
            f"=== ОЧИЩЕННЫЙ ТЕКСТ ===\n{cleaned_text}\n\n"
            f"=== ПЛАН JSON ===\n{plan_json}\n\n"
            f"=== РЕФЕРЕНС ===\n{reference}\n"
        )

    @staticmethod
    def latex_user_prompt(draft_text: str, plan: ManualPlan, reference: str) -> str:
        return (
            "Ниже даны markdown-черновик, план и референс.\n"
            "Верни только полный компилируемый LaTeX (pdfLaTeX).\n"
            "Обязательны: \\documentclass, \\begin{document}, \\end{document}.\n\n"
            f"=== MARKDOWN ===\n{draft_text}\n\n"
            f"=== ПЛАН JSON ===\n{plan.model_dump_json(indent=2, ensure_ascii=False)}\n\n"
            f"=== РЕФЕРЕНС ===\n{reference}\n"
        )

    @staticmethod
    def latex_fix_user_prompt(tex_code: str) -> str:
        return (
            "Ниже дан LaTeX-код. Исправь явные синтаксические проблемы, "
            "сохрани содержание и верни только полный исправленный .tex файл.\n\n"
            f"=== LATEX ===\n{tex_code}\n"
        )
