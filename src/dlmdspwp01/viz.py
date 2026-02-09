from __future__ import annotations

from typing import List
import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import column


class Visualizer:
    """Bokeh visualizations for training, ideal and mapped test points."""

    def __init__(self, x_col: str = "x"):
        self.x_col = x_col

    def plot(
        self,
        train_df: pd.DataFrame,
        ideal_df: pd.DataFrame,
        selected,
        mapped_df: pd.DataFrame,
        out_html: str,
        title: str = "DLMDSPWP01 – Ideal Function Mapping",
    ) -> None:
        p = figure(title=title, x_axis_label=self.x_col, y_axis_label="y", width=1000, height=600)
        p.add_tools(HoverTool(tooltips=[("x", "@x"), ("y", "@y"), ("series", "@series")]))

        # Training series
        for col in [c for c in train_df.columns if c != self.x_col]:
            src = ColumnDataSource({"x": train_df[self.x_col], "y": train_df[col], "series": [col]*len(train_df)})
            p.circle("x", "y", source=src, size=4, legend_label=f"train: {col}", alpha=0.5)

        # Selected ideal functions
        for s in selected:
            src = ColumnDataSource({"x": ideal_df[self.x_col], "y": ideal_df[s.ideal_col], "series": [s.ideal_col]*len(ideal_df)})
            p.line("x", "y", source=src, line_width=2, legend_label=f"ideal: {s.ideal_col}", alpha=0.8)

        # Mapped test points
        if mapped_df is not None and not mapped_df.empty:
            src = ColumnDataSource({"x": mapped_df[self.x_col], "y": mapped_df["y"], "series": mapped_df["ideal_func"]})
            p.triangle("x", "y", source=src, size=7, legend_label="mapped test", alpha=0.8)

        p.legend.click_policy = "hide"
        output_file(out_html)
        save(p)
