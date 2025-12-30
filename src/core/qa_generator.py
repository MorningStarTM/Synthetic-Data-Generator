# src/core/qa_generator.py

import os
import json
import dspy
from typing import Dict, Any, Optional
from src.core.providers.dspy_provider import DspyProvider
from src.utils.logger import logger
from src.utils.utils import save_qa_from_completions, qa_to_json
from src.eval.information import WeightedInformationCoverageMetric
from src.eval.similarity import QADiversityMetric
from src.utils.utils import load_topic_docs_from_dir
from src.optimizers.metric_optimizer import composite_metric
from src.core.template_registry import TemplateRegistry




class QAGenerator:
    def __init__(self, config: Dict[str, Any]):
        """
        High-level wrapper around DspyProvider for QA generation.

        Expected config shape (example):

        {
            "provider": {
                "model": "ollama/gemma3:4b",
                "api_key": "...",                     # if needed
                "lm_kwargs": {"max_tokens": 4096},
                "qa_template_path": "src/templates/qa_template.txt"
            },
            "context_dir": "src/context",
            "default_num_questions": 10
        }
        """
        self.config: Dict[str, Any] = config or {}

        # Where context .txt files live (user uploads)
        self.context_dir: str = self.config.get("context_dir", "src/context")

        # Provider-specific config (fallback: treat whole config as provider config)
        provider_config: Dict[str, Any] = self.config.get("provider", self.config)

        # Default number of Q&A pairs if not specified at call-time
        self.default_num_questions: int = self.config.get(
            "default_num_questions", 10
        )

        # Initialize DSPy-backed provider
        self.provider = DspyProvider(provider_config)
        self.registry = TemplateRegistry(config=config)

        self.div_metric = QADiversityMetric(embedding_model=self.config.get("embedding_model","all-MiniLM-L6-v2"))
        self.info_metric = WeightedInformationCoverageMetric(embedding_model=self.config.get("embedding_model","all-MiniLM-L6-v2"))


    def generate(
        self,
        context_filename: str,
        trainset:dspy.Example,
        num_questions: Optional[int] = None,
    ) -> str:
        """
        Load context from a .txt file and generate QA JSON.

        Args:
            context_filename: Name of the .txt file inside `context_dir`
                              (e.g. "agency_123_context.txt").
            num_questions: Optional override. If None, uses default_num_questions.

        Returns:
            JSON string of Q&A pairs produced by DspyProvider.generate().
        """
        step = 0
        # Resolve full path to context file
        context_path = os.path.join(self.context_dir, context_filename)

        if not os.path.exists(context_path):
            logger.error(f"Context file not found: {context_path}")
            raise FileNotFoundError(f"Context file not found: {context_path}")

        # Read context text
        with open(context_path, "r", encoding="utf-8") as f:
            context_text = f.read()

        # Decide how many questions to ask for
        n_questions = num_questions or self.default_num_questions

        # Options passed into DspyProvider.generate()
        options = {"num_questions": n_questions}

        logger.info(
            f"Generating Q&A from context file='{context_path}' "
            f"with num_questions={n_questions} using model={self.provider.model_name}"
        )

        # Delegate to DspyProvider
        qa_json = self.provider.generate(context=context_text, options=options)
        #save_qa_from_completions(qa_json.completions)
        
        qa2json = qa_to_json(qa_json.completions)
        gen_text = json.dumps(qa2json, ensure_ascii=False)
        
        topic_docs = load_topic_docs_from_dir(self.config['context_dir'])
        print(topic_docs)
        # Information coverage
        info_score, _ = self.info_metric(gen_text, topic_docs)

        # Diversity check
        div_score, _ = self.div_metric(qa2json)

        if info_score < 0.65 or div_score < 0.5:
            optimizer = dspy.SIMBA(metric=composite_metric, bsize=2, max_steps=2)
            optimized_program = optimizer.compile(self.provider.predictor, trainset=trainset)
            self.registry.save_prompt_template(
                name="qa_optimized_prompt",
                content=optimized_program.signature.instructions,
                category="optimized",     # stored in prompts/optimized/
                use_user_dir=False,        # user_configs/prompts/optimized/...
                add_timestamp=True,       # filename includes timestamp
            )


        
        """if step != 2:
            self.generate()"""


        return qa_json, optimized_program
