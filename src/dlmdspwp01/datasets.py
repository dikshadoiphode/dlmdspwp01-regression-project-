from __future__ import annotations

from dataclasses import dataclass
from typing import List
import pandas as pd

from .exceptions import DataValidationError


@dataclass
class Dataset:
    """Base dataset class.

    Attributes
    ----------
    path:
        Path to the CSV file.
    x_col:
        Name of the x column.
    """
    path: str
    x_col: str = "x"

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        self.validate(df)
        return df

    def validate(self, df: pd.DataFrame) -> None:
        if self.x_col not in df.columns:
            raise DataValidationError(f"Missing required x column '{self.x_col}' in {self.path}")


@dataclass
class TrainingDataset(Dataset):
    """Training dataset: expects x plus 4 y-columns (training functions)."""
    def y_cols(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c != self.x_col]

    def validate(self, df: pd.DataFrame) -> None:
        super().validate(df)
        y_cols = self.y_cols(df)
        if len(y_cols) != 4:
            raise DataValidationError(
                f"Training dataset must contain 4 training function columns (found {len(y_cols)}): {y_cols}"
            )


@dataclass
class IdealDataset(Dataset):
    """Ideal function dataset: expects x plus 50 y-columns (ideal functions)."""
    def y_cols(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c != self.x_col]

    def validate(self, df: pd.DataFrame) -> None:
        super().validate(df)
        y_cols = self.y_cols(df)
        if len(y_cols) != 50:
            raise DataValidationError(
                f"Ideal dataset must contain 50 ideal function columns (found {len(y_cols)}): first 10 = {y_cols[:10]}"
            )


@dataclass
class TestDataset(Dataset):
    """Test dataset: supports x plus one or more y-columns.

    The IU task describes a single Y column. If multiple columns exist, each is treated as a separate test series.
    """

    def y_cols(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c != self.x_col]

    def validate(self, df: pd.DataFrame) -> None:
        super().validate(df)
        y_cols = self.y_cols(df)
        if len(y_cols) < 1:
            raise DataValidationError("Test dataset must contain at least one Y column.")
