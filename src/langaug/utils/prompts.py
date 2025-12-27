import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template
from pydantic import BaseModel

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "data" / "prompts"


class PromptLoader:
    _env: Environment | None = None

    @classmethod
    def _get_env(cls) -> Environment:
        if cls._env is None:
            cls._env = Environment(loader=FileSystemLoader(PROMPTS_DIR), autoescape=False)
        return cls._env

    @classmethod
    def load(cls, prompt: str) -> Template:
        if prompt.endswith(".jinja2"):
            return cls._get_env().get_template(prompt)

        prompt_file = f"{prompt}.jinja2"
        if (PROMPTS_DIR / prompt_file).exists():
            return cls._get_env().get_template(prompt_file)

        return Template(prompt)

    @classmethod
    def render(cls, prompt: str, context: BaseModel | dict) -> str:
        template = cls.load(prompt)
        if isinstance(context, BaseModel):
            context = context.model_dump()
        rendered = template.render(**context)
        logger.debug("Rendered prompt (%d chars)", len(rendered))
        return rendered
