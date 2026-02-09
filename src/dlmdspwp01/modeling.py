from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
import pandas as pd
import numpy as np

from .exceptions import MappingNotPossible


@dataclass(frozen=True)
class SelectedIdeal:
    train_col: str
    ideal_col: str
    sse: float
    max_dev: float


class IdealFunctionSelector:
    """Select the best ideal function for each training function using SSE (least squares)."""

    def __init__(self, x_col: str = "x"):
        self.x_col = x_col

    def select(self, train_df: pd.DataFrame, ideal_df: pd.DataFrame) -> List[SelectedIdeal]:
        train_cols = [c for c in train_df.columns if c != self.x_col]
        ideal_cols = [c for c in ideal_df.columns if c != self.x_col]

        
        merged = pd.merge(train_df, ideal_df, on=self.x_col, how="inner", suffixes=("", ""))
        if merged.empty:
            raise ValueError("No overlapping x-values between training and ideal datasets.")

        selected: List[SelectedIdeal] = []
        for tcol in train_cols:
            best: Optional[SelectedIdeal] = None
            tvals = merged[tcol].to_numpy(dtype=float)

            for icol in ideal_cols:
                ivals = merged[icol].to_numpy(dtype=float)
                diff = tvals - ivals
                sse = float(np.sum(diff ** 2))
                max_dev = float(np.max(np.abs(diff)))
                cand = SelectedIdeal(train_col=tcol, ideal_col=icol, sse=sse, max_dev=max_dev)
                if best is None or cand.sse < best.sse:
                    best = cand

            assert best is not None
            selected.append(best)

        return selected


class TestDataMapper:
    """Map test points to selected ideal functions using the √2 rule."""

    def __init__(self, x_col: str = "x"):
        self.x_col = x_col

    def map_points(
        self,
        test_df: pd.DataFrame,
        ideal_df: pd.DataFrame,
        selected: List[SelectedIdeal],
        sqrt_factor: float = math.sqrt(2),
    ) -> pd.DataFrame:
        ideal_lookup = ideal_df.set_index(self.x_col)
        test_cols = [c for c in test_df.columns if c != self.x_col]

        
        sel_by_ideal: Dict[str, SelectedIdeal] = {s.ideal_col: s for s in selected}

        rows = []
        for _, r in test_df.iterrows():
            x = r[self.x_col]
            if x not in ideal_lookup.index:
                continue  
            ideal_row = ideal_lookup.loc[x]

            for tcol in test_cols:
                y = float(r[tcol])
                best_match: Optional[Tuple[str, float, float]] = None  
                for s in selected:
                    y_hat = float(ideal_row[s.ideal_col])
                    delta = abs(y - y_hat)
                    threshold = s.max_dev * sqrt_factor
                    if delta <= threshold:
                        if best_match is None or delta < best_match[1]:
                            best_match = (s.ideal_col, delta, threshold)

                if best_match is not None:
                    ideal_col, delta, threshold = best_match
                    rows.append(
                        {
                            self.x_col: float(x),
                            "y": y,
                            "test_series": tcol,
                            "ideal_func": ideal_col,
                            "delta_y": float(delta),
                            "threshold": float(threshold),
                        }
                    )
                

        return pd.DataFrame(rows)
