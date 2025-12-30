import dspy
from src.core.template_registry import TemplateRegistry  
from src.core.qa_generator import QAGenerator


config = {
    "provider": {
        "model": "ollama/gemma3:4b",
        "lm_kwargs": {"max_tokens": 4096},
        "qa_template_path": "src\\template\\prompt_template.txt",
    },
    "context_dir": "src\\context",
    "default_num_questions": 2,
    "example_format":{'question':'{question}','answer':'{answer}'},
    "output_dir": "src\\output"

}


qa_gen = QAGenerator(config)
output_json = qa_gen.generate("physics.txt", num_sample=2, prompt_optimization=False)
print(output_json)