import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from loguru import logger


class LCVisualizer:
    @staticmethod
    def plot(results, save_path=None):
        plt.figure(figsize=(8, 6))
        x = results["train_sizes"]
        plt.plot(x, results["train_mean"], "o-", color="steelblue", label="Training Score", linewidth=2)
        plt.fill_between(x, np.array(results["train_mean"]) - np.array(results["train_std"]),
                         np.array(results["train_mean"]) + np.array(results["train_std"]), alpha=0.15, color="steelblue")
        plt.plot(x, results["val_mean"], "o-", color="crimson", label="Validation Score", linewidth=2)
        plt.fill_between(x, np.array(results["val_mean"]) - np.array(results["val_std"]),
                         np.array(results["val_mean"]) + np.array(results["val_std"]), alpha=0.15, color="crimson")
        plt.xlabel("Training Examples"); plt.ylabel("Accuracy")
        plt.title(f"Learning Curve (Gap={results['gap']:.3f})"); plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
