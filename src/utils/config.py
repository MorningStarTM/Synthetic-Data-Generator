from src.utils.utils import read_txt
from pathlib import Path

config = {
    "provider": {
        "model": "ollama/gemma3:4b",
        "lm_kwargs": {"max_tokens": 4096},
        "qa_template_path": "src\\template\\prompt_template.txt",
    },
    "context_dir": "src\\context",
    "default_num_samples": 2,
    "example_format":   read_txt(Path("src\context\sample_data.txt")),
    "output_dir": "output",
    "filename":"sythentic.csv",
    "additional_prompts":read_txt(Path("src/template/additional_element/additional_prompt.txt")),
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
                    }

}