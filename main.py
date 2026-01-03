import dspy
from src.core.template_registry import TemplateRegistry  
from src.core.qa_generator import QAGenerator
from src.utils.config import config
import pandas as pd



qa_gen = QAGenerator(config)
output = qa_gen.generate_sensorial_data("sample_data.txt", num_sample=2)
print(output)