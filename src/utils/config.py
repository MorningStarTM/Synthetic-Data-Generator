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