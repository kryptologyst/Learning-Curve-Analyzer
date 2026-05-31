import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import LearningCurveTool
from src.data import load_data


class TestLearningCurve:
    def test_compute(self):
        X, y, _ = load_data("wine")
        tool = LearningCurveTool(cv=3)
        results = tool.compute(X, y)
        assert len(results["train_sizes"]) > 0
        assert len(results["train_mean"]) == len(results["train_sizes"])
        assert results["val_mean"][-1] > 0
