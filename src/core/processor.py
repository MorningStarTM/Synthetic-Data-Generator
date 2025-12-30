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
        self.task_description: str = (
            self.config.get("task_description")
            or "Question and Answer"
        )

        self.language: str = (
            self.config.get("language")
            or "English"
        )

        self.example_format: str = (
            self.config.get("example_format")
            or "to match given task"
        )

        self.additional_prompts: str = (
            self.config.get("additional_prompts")
            or "no additional prompts"
        )

        self.language_prompt = self._read_language_prompt("src\\template\\language_prompt.txt")


    def _read_language_prompt(self, file_path: str) -> str:
        """Read language prompt from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            return ""
        
    
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
            "task_description": self.task_description,
            "language": self.language,
            "language_prompt": self.language_prompt,
            "example_format": self.example_format,
            "additional_prompts": self.additional_prompts,
        }

        return template_text.format(**values)
