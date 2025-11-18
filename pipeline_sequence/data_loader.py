import os
import pandas as pd
from typing import Union, List

class DataLoader:
 

    def __init__(self, file_path: str, required_columns: List[str] = ["problem"]):
        self.file_path = file_path
        self.required_columns = required_columns
        self.df = None

    def load(self) -> pd.DataFrame:
        """
        Load the dataset based on file extension.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found at: {self.file_path}")

        ext = os.path.splitext(self.file_path)[1].lower()
        if ext == ".csv":
            self.df = pd.read_csv(self.file_path, engine="python", on_bad_lines="warn")
        elif ext == ".json":
            self.df = pd.read_json(self.file_path)
        else:
            raise ValueError("Unsupported file type. Only CSV and JSON are supported.")

        self._validate_columns()
        return self.df

    def _validate_columns(self):
        """
        Ensure all required columns exist in the dataset.
        """
        missing = [col for col in self.required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def get_column(self, column_name: str) -> pd.Series:
        """
        Return a specific column from the dataset.
        """
        if self.df is None:
            raise RuntimeError("Data not loaded yet. Call load() first.")
        if column_name not in self.df.columns:
            raise ValueError(f"Column {column_name} not found in dataset.")
        return self.df[column_name]

    def preview(self, n: int = 5):
        """
        Print the first n rows of the dataset.
        """
        if self.df is None:
            raise RuntimeError("Data not loaded yet. Call load() first.")
        print(self.df.head(n))

    def save_processed(self, output_path: str):
        """
        Save the loaded (or processed) DataFrame to CSV.
        """
        if self.df is None:
            raise RuntimeError("Data not loaded yet. Nothing to save.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"✅ Saved processed dataset to {output_path}")

