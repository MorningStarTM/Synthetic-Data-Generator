# src/core/prompt_processor.py

import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Union, Optional
from typing import Dict, Any, Optional
from src.core.stats_extractor import TimeSeriesStatsExtractor, ExtractorConfig



cfg = ExtractorConfig(
        detail="lite",               # "lite" for LLM prompts, "full" for debugging
        include_categorical=False,    # set True if you want categorical stats too
        resample_rule=None,           # set e.g. "1D" if needed
        fill_method="ffill",
        compute_cross_corr=True,
        compute_cross_lagged=True,
    )


extractor = TimeSeriesStatsExtractor(cfg)

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
        self.example_format = self.csv_to_column_json_str(self.example_format, n=10)



    def csv_to_column_json(
            self,
            csv_path: Union[str, Path],
            *,
            n: int = 10,
            sep: str = ",",
            encoding: Optional[str] = None,
            keep_na: bool = False,
        ) -> Dict[str, list]:
        """
        Read a CSV and return a dict-of-lists JSON shape:
        { "col1": [v1, v2, ... up to n], "col2": [...], ... }

        Notes:
        - Takes the first `n` rows from the CSV.
        - Converts NaN/NaT to None by default (JSON-friendly).
        - Keeps native Python types when possible (int/float/bool/str).
        """
        csv_path = Path(csv_path)

        df = pd.read_csv(csv_path, sep=sep, encoding=encoding)
        df = df.head(n)

        if keep_na:
            # Keep NaN as-is (will fail json.dumps unless you allow NaN)
            out: Dict[str, list] = {c: df[c].tolist() for c in df.columns}
            return out

        # JSON-safe: NaN/NaT -> None
        df = df.where(pd.notnull(df), None)

        out: Dict[str, list] = {}
        for col in df.columns:
            vals = df[col].tolist()
            # Convert pandas Timestamp -> ISO string (JSON-friendly)
            vals = [
                (v.isoformat() if isinstance(v, pd.Timestamp) else v)
                for v in vals
            ]
            out[str(col)] = vals

        return out


    def csv_to_column_json_str(
            self,
            csv_path: Union[str, Path],
            *,
            n: int = 10,
            indent: int = 2,
            **kwargs: Any,
        ) -> str:
        """Same as csv_to_column_json, but returns a JSON string."""
        data = self.csv_to_column_json(csv_path, n=n, **kwargs)
        return json.dumps(data, ensure_ascii=False, indent=indent)



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

        if self.config['task_description'] == "time-series":
            stats = extractor.from_csv(self.config['time_series_path'])
            stats_json = extractor.to_prompt_json(stats)
            ts_prompt_text = self._read_prompt(self.config['time_series_data_path'])
            additional_prompt = self.render_additional_prompts(
                                                        ts_prompt_text,
                                                        strict=True,
                                                        stats_placeholder=stats_json
                                                    )
        else:
            additional_prompt = self.additional_prompts

        values = {
            "num_samples": n,
            "task_description": self.task_description,
            "language": self.language,
            "language_prompt": self.language_prompt,
            "example_format": self.example_format,
            "additional_prompts": additional_prompt,
            **kwargs,  # allow extra placeholders for the main template too
        }

        return template_text.format(**values)
