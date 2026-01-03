import os
import csv
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import pandas as pd
from src.utils.logger import logger
import json

class Dumper:
    """
    Handles saving data to CSV files with validation and error handling.
    
    This class provides utilities for:
    - Saving dictionaries to CSV (row or column oriented)
    - Saving pandas DataFrames to CSV
    - Validating data before saving
    - Creating necessary directories
    """

    def __init__(self, config):
        """
        Initialize the Dumper with an output directory.

        Args:
            output_dir: Directory where CSV files will be saved.
                       Defaults to "output".
        """
        self.config = config
        self.output_dir = Path(self.config['output_dir'])
        self._ensure_output_dir_exists()

    def _ensure_output_dir_exists(self) -> None:
        """Create output directory if it doesn't exist."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory ready: {self.output_dir.resolve()}")
        except Exception as e:
            logger.error(f"Failed to create output directory {self.output_dir}: {e}")
            raise
    

    def _preprocess_llm_output(self, data: Union[str, Dict, List]) -> Union[Dict, List]:
        """
        Preprocess LLM output to convert string representations to actual dicts/lists.
        
        Handles cases where LLM returns:
        - String representation of list of dicts: "[{...}, {...}]"
        - String representation of dict: "{...}"
        - Already parsed dict/list objects
        
        Args:
            data: Raw LLM output (string or already parsed)
        
        Returns:
            Union[Dict, List]: Parsed dictionary or list of dictionaries
        
        Raises:
            ValueError: If data cannot be parsed
        """
        import ast
        
        # If already a dict or list, return as-is
        if isinstance(data, (dict, list)):
            return data
        
        if not isinstance(data, str):
            raise ValueError(f"Expected string, dict, or list. Got {type(data)}")
        
        data = data.strip()
        
        # Try JSON first
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            logger.debug("JSON parsing failed, trying ast.literal_eval")
        
        # Try Python literal eval (handles single quotes, etc.)
        try:
            return ast.literal_eval(data)
        except (ValueError, SyntaxError) as e:
            logger.error(f"Failed to parse LLM output: {e}")
            raise ValueError(
                f"Cannot parse LLM output as JSON or Python literal. "
                f"Error: {e}\nData sample: {data[:200]}..."
            ) from e
    

    def save_dict_to_csv(
                self,
                data: Union[str, Dict[str, Any], List[Dict[str, Any]]],
                filename: str,
                orientation: Optional[str] = None,
            ) -> str:
        """
        Save dictionary/list data to CSV file.
        
        Automatically detects orientation if not specified.
        Preprocesses string LLM output.

        Args:
            data: Data to save (string, dict, or list of dicts)
            filename: CSV filename (with or without .csv extension)
            orientation: "column", "row", or None for auto-detection

        Returns:
            str: Absolute path to saved CSV file
        
        Raises:
            ValueError: If data format is invalid
        """
        if not filename:
            raise ValueError("filename cannot be empty")

        # Preprocess: convert string LLM output to dict/list
        try:
            processed_data = self._preprocess_llm_output(data)
        except ValueError as e:
            logger.error(f"Preprocessing failed for {filename}: {e}")
            raise

        # Ensure .csv extension
        if not filename.endswith(".csv"):
            filename = f"{filename}.csv"

        file_path = self.output_dir / filename

        try:
            # Auto-detect orientation if not specified
            if orientation is None:
                orientation = self._detect_orientation(processed_data)
                logger.info(f"Auto-detected orientation: {orientation}")

            if orientation == "column":
                self._save_column_oriented(processed_data, file_path)
            elif orientation == "row":
                self._save_row_oriented(processed_data, file_path)
            else:
                raise ValueError(f"Invalid orientation: {orientation}. Must be 'column' or 'row'")

            logger.info(f"Successfully saved data to {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to save CSV to {file_path}: {e}")
            raise


    def _detect_orientation(self, data: Union[Dict, List]) -> str:
        """
        Auto-detect data orientation.
        
        Returns:
            str: "row" for list of dicts, "column" for dict of lists
        """
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return "row"
        elif isinstance(data, dict):
            return "column"
        else:
            raise ValueError("Cannot auto-detect orientation. Data must be dict or list of dicts")

    def _save_column_oriented(self, data: Dict[str, List[Any]], file_path: Path) -> None:
        """
        Save column-oriented dictionary as CSV.

        Expected format: {"col1": [v1, v2, ...], "col2": [v1, v2, ...]}
        """
        if not isinstance(data, dict):
            raise ValueError("Column-oriented data must be a dictionary")

        # Validate all values are lists and have consistent length
        if not data:
            raise ValueError("Data dictionary is empty")

        lengths = {k: len(v) if isinstance(v, list) else 1 for k, v in data.items()}
        unique_lengths = set(lengths.values())

        if len(unique_lengths) > 1:
            raise ValueError(
                f"Inconsistent column lengths: {lengths}. "
                "All columns must have the same length."
            )

        # Convert to pandas DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False, encoding="utf-8")

    def _save_row_oriented(self, data: List[Dict[str, Any]], file_path: Path) -> None:
        """
        Save row-oriented data (list of dicts) as CSV.

        Expected format: [{"col1": v1, "col2": v2}, ...]
        """
        if not isinstance(data, list):
            raise ValueError("Row-oriented data must be a list of dictionaries")

        if not data:
            raise ValueError("Data list is empty")

        if not all(isinstance(row, dict) for row in data):
            raise ValueError("All items in row-oriented data must be dictionaries")

        # Convert to pandas DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False, encoding="utf-8")

    def save_dataframe_to_csv(
        self,
        df: pd.DataFrame,
        filename: str,
        include_index: bool = False,
    ) -> str:
        """
        Save a pandas DataFrame to CSV file.

        Args:
            df: pandas DataFrame to save.
            filename: Name of the CSV file (with or without .csv extension).
            include_index: Whether to include DataFrame index. Defaults to False.

        Returns:
            str: Absolute path to the saved CSV file.

        Raises:
            ValueError: If DataFrame is empty or invalid.
            IOError: If file cannot be written.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if df.empty:
            raise ValueError("Cannot save empty DataFrame")

        if not filename:
            raise ValueError("filename cannot be empty")

        # Ensure .csv extension
        if not filename.endswith(".csv"):
            filename = f"{filename}.csv"

        file_path = self.output_dir / filename

        try:
            df.to_csv(file_path, index=include_index, encoding="utf-8")
            logger.info(f"Successfully saved DataFrame to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save DataFrame to {file_path}: {e}")
            raise

    def set_output_dir(self, output_dir: str) -> None:
        """
        Change the output directory.

        Args:
            output_dir: New output directory path.
        """
        self.output_dir = Path(output_dir)
        self._ensure_output_dir_exists()

    def get_output_dir(self) -> str:
        """
        Get the current output directory.

        Returns:
            str: Absolute path to the output directory.
        """
        return str(self.output_dir.resolve())