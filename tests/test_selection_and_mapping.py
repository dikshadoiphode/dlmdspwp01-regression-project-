import math
import pandas as pd

from dlmdspwp01.modeling import IdealFunctionSelector, TestDataMapper
from dlmdspwp01.datasets import TrainingDataset, IdealDataset, TestDataset


def test_select_returns_four_functions():
    train = TrainingDataset("data/train.csv").load()
    ideal = IdealDataset("data/ideal.csv").load()
    sel = IdealFunctionSelector().select(train, ideal)
    assert len(sel) == 4
    assert len({s.train_col for s in sel}) == 4
    assert all(s.sse >= 0 for s in sel)
    assert all(s.max_dev >= 0 for s in sel)


def test_mapping_respects_threshold():
    train = TrainingDataset("data/train.csv").load()
    ideal = IdealDataset("data/ideal.csv").load()
    test = TestDataset("data/test.csv").load()

    sel = IdealFunctionSelector().select(train, ideal)
    mapped = TestDataMapper().map_points(test, ideal, sel, sqrt_factor=math.sqrt(2))


    assert not mapped.empty
    assert (mapped["delta_y"] <= mapped["threshold"] + 1e-12).all()
