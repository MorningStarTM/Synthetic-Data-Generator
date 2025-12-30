import os
from typing import Dict, Any, Optional
import pandas as pd
import PyPDF2


class ContextManager:
    def __init__(self, config: Dict[str, Any]):
        """
        Manages context files for the application.

        Expected config shape (example):

        {
            "context_dir": "src/context"
        }
        """
        self.config: Dict[str, Any] = config or {}

        # Where context .txt files live (user uploads)
        self.context_dir: str = self.config.get("context_dir", "src/context")

    def read_context(self, context_filename: str) -> Optional[str]:
        """
        Reads the content of a context file.

        Supports: .txt, .csv, .xlsx, .pdf

        Args:
            context_filename (str): The name of the context file to read.

        Returns:
            Optional[str]: The content of the context file, or None if the file does not exist.
        """
        context_path = os.path.join(self.context_dir, context_filename)
        if not os.path.isfile(context_path):
            return None

        file_ext = os.path.splitext(context_filename)[1].lower()

        try:
            if file_ext == '.txt':
                return self._read_txt(context_path)
            elif file_ext == '.csv':
                return self._read_csv(context_path)
            elif file_ext == '.xlsx':
                return self._read_xlsx(context_path)
            elif file_ext == '.pdf':
                return self._read_pdf(context_path)
            else:
                return None
        except Exception as e:
            print(f"Error reading file {context_filename}: {e}")
            return None

    def _read_txt(self, file_path: str) -> str:
        """Read .txt file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def _read_csv(self, file_path: str) -> str:
        """Read .csv file and convert to string."""
        df = pd.read_csv(file_path)
        return df.to_string()

    def _read_xlsx(self, file_path: str) -> str:
        """Read .xlsx file and convert to string."""
        df = pd.read_excel(file_path)
        return df.to_string()

    def _read_pdf(self, file_path: str) -> str:
        """Read .pdf file and extract text."""
        text = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        return '\n'.join(text)