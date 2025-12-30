from src.eval.information import WeightedInformationCoverageMetric
from src.eval.similarity import QADiversityMetric
from src.utils.config import config
import dspy
import json
from src.utils.utils import qa_to_json
from src.utils.utils import load_topic_docs_from_dir



div_metric = QADiversityMetric(embedding_model=config.get("embedding_model","all-MiniLM-L6-v2"))
info_metric = WeightedInformationCoverageMetric(embedding_model=config.get("embedding_model","all-MiniLM-L6-v2"))


def composite_metric(example, prediction, trace=None):
   qa2json = qa_to_json(prediction.completions)
   gen_text = json.dumps(qa2json, ensure_ascii=False)

   topic_docs = load_topic_docs_from_dir('src\\context')
   score1, _ = div_metric(qa2json)

   score2, _ = info_metric(gen_text, topic_docs)
   score = (score1 + score2) / 2.0
   feedback = f"You scored {score1}/1.0 and {score2}/1.0 on diversity and information coverage, respectively"
   return dspy.Prediction(score=score, feedback=feedback)