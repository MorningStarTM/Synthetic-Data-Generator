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

        # Priority: explicit default_num_samples, then num_samples, else 10
        self.default_num_samples: int = (
            self.config.get("default_num_samples")
            or self.config.get("num_samples")
            or 2
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

        self.language_prompt = self._read_prompt("src\\template\\language_prompt.txt")
        self.additional_prompts = self._read_prompt(self.config['additional_prompt'])


    def _read_prompt(self, file_path: str) -> str:
        """Read language prompt from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            return ""


    def render_additional_prompts(
        self,
        additional_template: Optional[str] = None,
        *,
        strict: bool = True,
        **extra_vars: Any,
    ) -> str:
        """
        Render placeholders inside the additional prompt template, then return the final string.

        Usage:
            rendered_add = self.render_additional_prompts(
                stats_placeholder=stats_json,
                time_range="2022-01-01 to 2024-12-31",
                frequency="monthly",
            )
            # then pass `rendered_add` into values["additional_prompts"]

        Args:
            additional_template: Optional override template. If None, uses self.additional_prompts.
            strict: If True, raises KeyError if any placeholder is missing.
                    If False, leaves unknown placeholders unchanged.
            extra_vars: placeholder values, e.g. stats_placeholder="...", etc.
        """
        template = additional_template if additional_template is not None else (self.additional_prompts or "")

        if strict:
            # Fail fast if any placeholder is missing
            return template.format(**extra_vars)

        # Non-strict: leave unknown placeholders as-is
        class _SafeDict(dict):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"

        return template.format_map(_SafeDict(extra_vars))



    
    def render(
            self,
            template_text: str,
            num_samples: Optional[int] = None,
            *args: Any,
            **kwargs: Any,
        ) -> str:
        n = num_samples if num_samples is not None else self.default_num_samples

        # Convention:
        # args[0] -> additional_prompt template override (str)   (optional)
        # args[1] -> additional_prompt_vars (dict)              (optional)
        additional_prompt = args[0] if len(args) > 0 else None
        additional_prompt_vars = args[1] if len(args) > 1 else None

        add_tpl = additional_prompt if additional_prompt is not None else self.additional_prompts
        add_vars = additional_prompt_vars if isinstance(additional_prompt_vars, dict) else {}

        rendered_additional = self.render_additional_prompts(
            additional_template=add_tpl,
            strict=False,
            **add_vars,
        )

        values = {
            "num_samples": n,
            "task_description": self.task_description,
            "language": self.language,
            "language_prompt": self.language_prompt,
            "example_format": self.example_format,
            "additional_prompts": rendered_additional,
            **kwargs,  # allow extra placeholders for the main template too
        }

        return template_text.format(**values)
