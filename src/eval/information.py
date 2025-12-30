import re
from typing import List, Tuple, Dict, Optional
import torch
from sentence_transformers import SentenceTransformer, util
try:
    from nltk.tokenize import sent_tokenize
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False


def _safe_sent_tokenize(text: str) -> List[str]:
    """Robust sentence splitter with NLTK if available, else regex fallback."""
    text = text.strip()
    if not text:
        return []
    if _HAS_NLTK:
        try:
            sents = sent_tokenize(text)
            return [s.strip() for s in sents if s.strip()]
        except LookupError:
            pass
    # Very simple fallback
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]


class WeightedInformationCoverageMetric:
    """
    Measures how well generated QA text covers the information in topic/context docs.

    Usage:
        metric = WeightedInformationCoverageMetric("all-MiniLM-L6-v2")
        score, details = metric(generated_qa_text, topic_docs)

    Returns:
        score: float in [0, 1] (higher = better coverage)
        details: dict with extra info (coverage_rate, etc.)
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        shared_model: Optional[SentenceTransformer] = None,
    ):
        if shared_model is not None:
            self.model = shared_model
        else:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(embedding_model, device=device)

    def __call__(self, generated_text: str, topic_docs: List[str]) -> Tuple[float, Dict]:
      # Break context into "units" (sentences)
      source_units: List[str] = []
      for doc in topic_docs:
          source_units.extend(_safe_sent_tokenize(doc))

      # Break generated QA into sentences too
      gen_units = _safe_sent_tokenize(generated_text)

      # Encode & normalize (this is where embeddings happen)
      source_emb = self.model.encode(
          source_units, convert_to_tensor=True, normalize_embeddings=True
      )
      gen_emb = self.model.encode(
          gen_units, convert_to_tensor=True, normalize_embeddings=True
      )

      # Cosine similarity: (num_gen, num_source)
      sim_matrix = util.cos_sim(gen_emb, source_emb)  # already in [-1, 1]

      # For each source unit, what is the best matching QA sentence?
      # sim_matrix: rows = gen_units, cols = source_units
      max_sim_per_source = torch.max(sim_matrix, dim=0).values  # (num_source,)

      # Map cosine [-1, 1] -> [0, 1]
      max_sim_per_source_01 = (max_sim_per_source + 1.0) / 2.0
      coverage_score = float(max_sim_per_source_01.mean().item())

      # "Coverage rate": fraction of context units that are well-covered
      threshold = 0.4  # you can tune this
      covered_mask = max_sim_per_source_01 >= threshold
      coverage_rate = float(covered_mask.float().mean().item())

      info = {
          "mean_max_similarity_01": coverage_score,
          "coverage_rate": coverage_rate,
          "num_source_units": len(source_units),
          "num_generated_units": len(gen_units),
      }

      return coverage_score, info
