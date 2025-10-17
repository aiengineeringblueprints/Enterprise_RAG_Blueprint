# src/prompt_manager.py
from enum import Enum
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

class PromptKey(str, Enum):
    SUMMARY  = "rag_summary"                # makes summary of the content
    SOURCE   = "rag_source"                 # acts as a rag system with citing sources
    CHECK  = "rag_check"                    # checks the original question and answer to it, output "yes" or "no"
    CHECK_EXTENDED = "rag_check_extended"   # checks the original question and answer to it, output explanation
    WEEKLY_REPORT = "rag_weekly_report"     # first try for a weekly report, TODO: test on real data
    TEST_PROMT_TEMPLATE = "test_prompt_template"  # test prompt template - template for all other promts?
    TEST_PROMT_TEMPLATE_QA = "test_prompt_template_QA"  # test prompt template for q and a


class PromptManager:
    def __init__(self, template_dir: str = None):
        base = Path(__file__).parent.parent / "prompts"
        self.template_dir = template_dir or str(base)
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(),  # keine HTML‑Escapes
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def load_template(self, key: PromptKey) -> str:
        """Load the raw template string (with Jinja placeholders)."""
        source, _, _ = self.env.loader.get_source(self.env, f"{key.value}.j2")
        return source

    def render(self, key: PromptKey, **variables) -> str:
        """Render the template with the provided variables."""
        tpl = self.env.get_template(f"{key.value}.j2")
        return tpl.render(**variables)
