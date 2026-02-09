from __future__ import annotations

from typing import Optional
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from .exceptions import DatabaseError


class DatabaseManager:
    """Creates and manages the SQLite database using SQLAlchemy."""

    def __init__(self, sqlite_path: str):
        self.sqlite_path = sqlite_path
        self._engine: Optional[Engine] = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            try:
                self._engine = create_engine(f"sqlite:///{self.sqlite_path}")
            except Exception as e:
                raise DatabaseError(str(e)) from e
        return self._engine

    def write_table(self, df: pd.DataFrame, table_name: str) -> None:
        try:
            df.to_sql(table_name, self.engine, if_exists="replace", index=False)
        except Exception as e:
            raise DatabaseError(f"Failed to write table '{table_name}': {e}") from e

    def read_table(self, table_name: str) -> pd.DataFrame:
        try:
            return pd.read_sql_table(table_name, self.engine)
        except Exception as e:
            raise DatabaseError(f"Failed to read table '{table_name}': {e}") from e
