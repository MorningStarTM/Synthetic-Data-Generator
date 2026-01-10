# src/core/dspy_qa.py

from pathlib import Path
from typing import Type
from src.core.processor import QAPromptProcessor
import dspy
from src.utils.config import config
from src.core.template_registry import TemplateRegistry
from src.core.stats_extractor import TimeSeriesStatsExtractor, ExtractorConfig
from src.utils.logger import logger



cfg = ExtractorConfig(
        detail="lite",               # "lite" for LLM prompts, "full" for debugging
        include_categorical=False,    # set True if you want categorical stats too
        resample_rule=None,           # set e.g. "1D" if needed
        fill_method="ffill",
        compute_cross_corr=True,
        compute_cross_lagged=True,
    )


processor = QAPromptProcessor(config)
extractor = TimeSeriesStatsExtractor(cfg)



def load_prompt_template(template_path: str) -> str:
    """Load a prompt template from a text file."""
    path = Path(template_path)
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def build_qa_signature_from_latest_template(
    registry: TemplateRegistry,
    category: str = None,
    name_prefix: str = None,
) -> Type[dspy.Signature]:
    """
    Build a DSPy Signature using the most recent optimized prompt template.
    """
    template_path = registry.get_latest_prompt_template_path(
        category=category,
        use_user_dir=False,
        name_prefix=name_prefix,
    )
    if config['task_description']:
        stats = extractor.from_csv(config['time_series_path'])
        stats_json = extractor.to_prompt_json(stats)
        ts_prompt_text = load_prompt_template(config['time_series_data_path'])
        additional_prompt = processor.render_additional_prompts(
                                                    ts_prompt_text,
                                                    strict=True,
                                                    stats_placeholder=stats_json
                                                )
    raw_prompt_text = load_prompt_template(template_path)
    prompt_text = processor.render(raw_prompt_text, None, additional_prompt)
    logger.info(f"prompt:\n  ===============================\n\n{prompt_text}\n\n==============================")


    class QAGenerationSignature(dspy.Signature):
        """This will be overwritten with the template content."""
        
        context = dspy.InputField(
            desc="input your conext to generate data"
        )
        
        answer: str = dspy.OutputField(
            desc="based on given task, return data as answer, "
        )

    # Inject your template as the docstring used by DSPy
    QAGenerationSignature.__doc__ = prompt_text

    return QAGenerationSignature
