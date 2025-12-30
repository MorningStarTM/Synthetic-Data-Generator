# src/core/prompt_processor.py

from typing import Dict, Any, Optional


class QAPromptProcessor:
    """
    Processes QA prompt templates by injecting config-driven values such as {num_question}.

    Example template:
        "You're Language Assistant to generate quality Question and Answer set
         for given Context. Generate {num_question} for given context."
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: main config dict.

            Expected keys (any of these, in this priority):
                - "default_num_questions"
                - "num_questions"
        """
        self.config = config or {}

        # Priority: explicit default_num_questions, then num_questions, else 10
        self.default_num_questions: int = (
            self.config.get("default_num_questions")
            or self.config.get("num_questions")
            or 10
        )

    def render(
        self,
        template_text: str,
        num_questions: Optional[int] = None,
    ) -> str:
        """
        Fill {num_question} in the given template text.

        Args:
            template_text: Raw template string containing `{num_question}`.
            num_questions: Optional override per call.
                           If None, uses value from config.

        Returns:
            Rendered template string with {num_question} replaced.
        """
        n = num_questions if num_questions is not None else self.default_num_questions

        # Extend this dict later if you add more placeholders
        values = {
            "num_question": n,
        }

        return template_text.format(**values)
