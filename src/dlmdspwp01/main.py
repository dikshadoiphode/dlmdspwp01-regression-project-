from __future__ import annotations

import argparse
import os
from pathlib import Path
import math

from .datasets import TrainingDataset, IdealDataset, TestDataset
from .db import DatabaseManager
from .modeling import IdealFunctionSelector, TestDataMapper
from .viz import Visualizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DLMDSPWP01 – Ideal function selection and test mapping")
    p.add_argument("--train", required=True, help="Path to training CSV (x + 4 columns)")
    p.add_argument("--ideal", required=True, help="Path to ideal functions CSV (x + 50 columns)")
    p.add_argument("--test", required=True, help="Path to test CSV (x + 1+ columns)")
    p.add_argument("--db", required=True, help="Output SQLite path")
    p.add_argument("--report", default="reports/mapping.html", help="Output HTML visualization path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(os.path.dirname(args.db) or ".").mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.report) or ".").mkdir(parents=True, exist_ok=True)

    train_df = TrainingDataset(args.train).load()
    ideal_df = IdealDataset(args.ideal).load()
    test_df = TestDataset(args.test).load()

    db = DatabaseManager(args.db)
    db.write_table(train_df, "training_data")
    db.write_table(ideal_df, "ideal_functions")

    selector = IdealFunctionSelector()
    selected = selector.select(train_df, ideal_df)

    # store selected mapping summary
    selected_df = __selected_to_df(selected)
    db.write_table(selected_df, "selected_ideal_functions")

    mapper = TestDataMapper()
    mapped_df = mapper.map_points(test_df, ideal_df, selected, sqrt_factor=math.sqrt(2))
    db.write_table(mapped_df, "mapped_test_data")

    viz = Visualizer()
    viz.plot(train_df, ideal_df, selected, mapped_df, args.report)

    print("Selected ideal functions:")
    print(selected_df.to_string(index=False))
    print(f"\nSQLite database written to: {args.db}")
    print(f"Visualization written to: {args.report}")


def __selected_to_df(selected):
    import pandas as pd
    return pd.DataFrame(
        [
            {"training_function": s.train_col, "ideal_function": s.ideal_col, "sse": s.sse, "max_deviation": s.max_dev}
            for s in selected
        ]
    )


if __name__ == "__main__":
    main()
