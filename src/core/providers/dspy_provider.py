import os
from typing import Dict, Any, Optional, List, Union
import dspy
from src.core.providers.model_provider import ModelProvider
from src.utils.logger import logger
from src.core.model_signature import build_qa_signature_from_latest_template
from src.core.template_registry import TemplateRegistry

class DspyProvider(ModelProvider):
    """
    ModelProvider implementation backed by a DSPy LM.

    This class:
      - Initializes a dspy.LM internally (e.g. "openai/gpt-4o-mini")
      - Implements `generate()` using that LM
      - Can be used anywhere a ModelProvider is expected
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the DspyProvider.

        Expected config keys:
            - model:       DSPy model identifier (e.g. "openai/gpt-4o-mini")
            - api_key:     API key for the backend (e.g. OpenAI key)
            - lm_kwargs:   Optional dict of extra kwargs for dspy.LM
                           (e.g. {"max_tokens": 2048})
        """
        config = config or {}

        # Model + API key
        self.model_name = config.get("model") or os.getenv(
            "DSPY_MODEL", "ollama/gemma3:4b"
        )
        api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY")

        # Extra LM kwargs: temperature, max_tokens, etc.
        lm_kwargs: Dict[str, Any] = config.get("lm_kwargs", {})
        if api_key is not None:
            lm_kwargs.setdefault("api_key", api_key)

        # Initialize underlying DSPy LM
        self.lm = dspy.LM(self.model_name, **lm_kwargs)
        self.registry = TemplateRegistry(config=config)

        # Optional: configure global DSPy default LM
        dspy.configure(lm=self.lm)

        self.signature = build_qa_signature_from_latest_template(registry=self.registry)#(config.get("qa_template_path", "src/templates/qa_template.txt"))
        self.predictor = dspy.Predict(signature=self.signature)

        # Call base init to store model_name
        super().__init__(model_name=self.model_name)

    def generate(
        self,
        context: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate text using DSPy via dspy.Predict over a simple Signature.

        `prompt` here is the full instruction; if you want more structured
        behavior (like context + num_questions + JSON), use a task-specific
        method (see `generate_qa_from_template` below).
        """
        options = options or {}

        try:
            # Build a Predict module on top of our FreeFormGeneration signature.
            # If you want per-call settings (like temperature), you can pass them
            # into dspy.settings or incorporate them into the template.
            

            # Call it with the instruction as input
            result = self.predictor(context=context)

            # Return the completion field
            return result

        except Exception as e:
            logger.error(f"DspyProvider.generate error: {e}")
            return ""

    def health_check(self) -> bool:
        """
        Basic health check: try a tiny no-op call.

        You can make this more sophisticated if needed.
        """
        try:
            _ = self.lm("ping", max_tokens=1)
            return True
        except Exception as e:
            logger.warning(f"DspyProvider health_check failed: {e}")
            return False
