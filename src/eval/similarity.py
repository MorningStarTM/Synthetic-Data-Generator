from itertools import combinations
from typing import Any

from typing import List, Tuple, Dict, Optional
import torch
from sentence_transformers import SentenceTransformer, util

class QADiversityMetric:
    """
    Measures *diversity* within a set of QA pairs using embeddings.

    Higher score = more diverse (less redundant).

    You can call it with:
        - a list of strings (already flattened)
        - OR a list of {"question": ..., "answer": ...} dicts
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

      
    def _qa_pairs_to_texts(self, qa_pairs: List[Any]) -> List[str]:
        """
        Normalizes input into a list of text strings representing each QA.

        Accepts:
            - list[str]
            - list[dict] with keys "question" and "answer"
        """
        if not qa_pairs:
            return []

        if isinstance(qa_pairs[0], str):
            return [t.strip() for t in qa_pairs if str(t).strip()]

        texts = []
        for item in qa_pairs:
            if not isinstance(item, dict):
                continue
            q = str(item.get("question", "")).strip()
            # a = str(item.get("answer", "")).strip()   # IGNORE answer for diversity

            if not q:
                continue
            texts.append(q)  # only question text
        return texts


    def __call__(self, qa_pairs: List[Any]) -> Tuple[float, Dict]:
        """
        Args:
            qa_pairs: list[str] OR list[{"question":..., "answer":...}, ...]

        Returns:
            (diversity_score, info_dict)
            diversity_score in [0, 1], higher = more diverse (less redundant).
        """
        texts = self._qa_pairs_to_texts(qa_pairs)
        n = len(texts)
        if n < 2:
            # Single QA can't be "redundant" with anything else
            return 0.5, {"reason": "not enough items", "num_items": n}

        emb = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

        # Pairwise cosine similarities
        sims = []
        for i, j in combinations(range(n), 2):
            sim_ij = util.cos_sim(emb[i], emb[j]).item()  # [-1, 1]
            sims.append(sim_ij)

        if not sims:
            return 0.5, {"reason": "no pairs", "num_items": n}

        sims = torch.tensor(sims)
        sims_01 = (sims + 1.0) / 2.0  # map to [0, 1]
        mean_sim_01 = float(sims_01.mean().item())

        # Diversity = 1 - avg similarity
        diversity_score = 1.0 - mean_sim_01

        info = {
            "mean_pairwise_similarity_01": mean_sim_01,
            "diversity_score": diversity_score,
            "num_items": n,
        }
        return diversity_score, info
