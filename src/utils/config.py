from src.utils.utils import read_txt
from pathlib import Path

config = {
    
    # "provider": {
    #     "hoster":"ollama",
    #     "model": "ollama/gemma3:4b",
    #     "lm_kwargs": {"max_tokens": 4096},
    #     "qa_template_path": "src\\template\\prompt_template.txt",
    # },

    "provider": {
        "hoster": "HF",
        "model": "meta-llama/Llama-3.1-8B-Instruct", #"meta-llama/Llama-3.1-8B-Instruct",
        "lm_kwargs": { "max_tokens": 2048, "temperature": 0.2 }
    },
    
    "context_dir": "src\\context",
    "default_num_samples": 2,
    "example_format":   Path("src\\context\\sample_data.txt"),
    "output_dir": "output",
    "filename":"sythentic.csv",
    "additional_prompts":read_txt(Path("src/template/additional_element/additional_prompt.txt")),
    "additional_prompt":Path("src\\template\\additional_element\\additional_prompt.txt"),
    'expected_schema':{
                        "PlayerID": "int",
                        "Age": "int",
                        "Gender": "str",
                        "Location": "str",
                        "GameGenre": "str",
                        "PlayTimeHours": "float",
                        "InGamePurchases": "int",
                        "GameDifficulty": "str",
                        "SessionsPerWeek": "int",
                        "AvgSessionDurationMinutes": "int",
                        "PlayerLevel": "int",
                        "AchievementsUnlocked": "int",
                        "EngagementLevel": "str",
                    },
    'task_description':'imbalance class data generation',
    'time_series_data_path':Path('src/template/additional_element/time_series_prompt.txt'),
    'time_series_path':Path("src\\context\\Month_Value.csv")

}