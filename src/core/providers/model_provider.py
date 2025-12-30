# src/core/model_provider.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from src.utils.logger import logger


class ModelProvider(ABC):
    """
    Abstract base class defining the interface for all model providers.
    """

    def __init__(self, model_name: str = ""):
        # Optional: store model name for logging / debugging
        self.model_name = model_name

    # ---- Core abstract API ----
    @abstractmethod
    def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the model provider is available."""
        pass

    # ---- Convenience / compatibility layer ----
    def predict(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Convenience method so this provider can be used in places that expect `.predict()`.
        Internally just calls `generate()`.
        """
        return self.generate(prompt, options)

    # ---- Batch helpers ----
    def generate_batch(
        self,
        prompt: str,
        num_samples: int,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generate multiple samples from the same prompt.

        Default implementation: call generate() multiple times (for Ollama).
        Cloud providers can override for efficient batch generation.
        """
        logger.info(
            f"Generating {num_samples} samples using default batch method (multiple API calls)"
        )
        results = []
        for i in range(num_samples):
            logger.info(f"Generating sample {i + 1}/{num_samples}")
            response = self.generate(prompt, options)
            results.append(response)
        return results

    def supports_batch_generation(self) -> bool:
        """Return True if provider supports efficient batch generation."""
        return False  # Default: False for Ollama-like providers
