from enum import Enum


class PipelineMode(str, Enum):
    AUDIO = "audio"
    TEXT_FILE = "text-file"


class StageName(str, Enum):
    CLEAN = "clean"
    PLAN_SOURCE = "plan_source"
    PLAN_JSON = "plan_json"
    DRAFT = "draft"
    LATEX = "latex"
    LATEX_FIX = "latex_fix"
