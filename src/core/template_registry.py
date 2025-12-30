from pathlib import Path
import os
from src.utils.logger import logger
import glob
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime



class TemplateRegistry:
    def __init__(self, config: Dict[str, Any]) -> None:
        dirs = config.get("directories", {})
        self.templates_dir = dirs.get("templates", "src")
        self.user_configs_dir = dirs.get("user_configs", "user_configs")

    def refresh(self) -> None:
        """
        Stub for whatever you already use to reload templates.
        Keep your existing implementation here.
        """
        pass

    def save_prompt_template(
        self,
        name: str,
        content: str,
        category: str = "optimized",
        use_user_dir: bool = False,
        add_timestamp: bool = True,
    ) -> str:
        """
        Save a prompt text as a .txt file into the prompts directory.

        Args:
            name: Base template name (without .txt).
            content: The prompt text to save.
            category: Optional subfolder under 'prompts' (e.g. 'optimized').
            use_user_dir: If True, save under user_configs/prompts, else templates/prompts.
            add_timestamp: If True, append a YYYYMMDD_HHMMSS timestamp to the filename.

        Returns:
            str: Absolute path to the saved template file.
        """
        base_dir = self.user_configs_dir if use_user_dir else self.templates_dir

        # Base: <base_dir>/prompts[/<category>]
        prompts_dir = Path(base_dir) / "template"
        if category:
            prompts_dir = prompts_dir / category

        os.makedirs(prompts_dir, exist_ok=True)

        # Sanitize base name (optional but safer)
        base_name = name.replace(" ", "_")

        if add_timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_{ts}.txt"
        else:
            # Ensure .txt extension
            filename = base_name if base_name.endswith(".txt") else f"{base_name}.txt"

        file_path = prompts_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Saved prompt template to {file_path}")


        return str(file_path)
    

    def get_latest_prompt_template_path(
        self,
        category: str = None,
        use_user_dir: bool = False,
        name_prefix: Optional[str] = None,
    ) -> str:
        """
        Find the most recent .txt prompt file and return its path.

        Args:
            category: Subfolder under 'prompts' to search (e.g. 'optimized').
            use_user_dir: If True, search under user_configs/prompts, else templates/prompts.
            name_prefix: Optional filename prefix filter
                         (e.g. 'qa_optimized_prompt' to only consider those files).

        Returns:
            str: Absolute path to the most recently modified matching .txt file.

        Raises:
            FileNotFoundError: If no matching prompt files are found.
        """
        base_dir = self.user_configs_dir if use_user_dir else self.templates_dir
        prompts_dir = Path(base_dir) / "template"
        if category:
            prompts_dir = prompts_dir / category

        if not prompts_dir.exists():
            raise FileNotFoundError(f"No prompts directory found at: {prompts_dir}")

        # Collect all .txt files
        candidates = list(prompts_dir.glob("*.txt"))
        if not candidates:
            raise FileNotFoundError(f"No .txt prompt files found in: {prompts_dir}")

        # Optionally filter by prefix
        if name_prefix is not None:
            candidates = [p for p in candidates if p.name.startswith(name_prefix)]
            if not candidates:
                raise FileNotFoundError(
                    f"No prompt files starting with '{name_prefix}' in: {prompts_dir}"
                )

        # Sort by modification time, newest first
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_path = candidates[0].resolve()

        logger.info(f"Latest prompt template selected: {latest_path}")
        return str(latest_path)
