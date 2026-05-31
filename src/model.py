import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from loguru import logger


class LearningCurveTool:
    def __init__(self, cv: int = 5, random_state: int = 42):
        self.cv = cv
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.train_sizes_: np.ndarray = None
        self.train_scores_: np.ndarray = None
        self.val_scores_: np.ndarray = None

    def compute(self, X, y, model=None, train_sizes=None) -> dict:
        X_scaled = self.scaler.fit_transform(X)
        if model is None:
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        self.train_sizes_, self.train_scores_, self.val_scores_ = learning_curve(
            model, X_scaled, y, cv=self.cv, n_jobs=-1,
            train_sizes=train_sizes, scoring="accuracy",
            random_state=self.random_state,
        )
        train_mean = self.train_scores_.mean(axis=1)
        train_std = self.train_scores_.std(axis=1)
        val_mean = self.val_scores_.mean(axis=1)
        val_std = self.val_scores_.std(axis=1)
        logger.info(
            f"Learning curve: {len(self.train_sizes_)} points, "
            f"final train={train_mean[-1]:.3f}, val={val_mean[-1]:.3f}"
        )
        return {
            "train_sizes": self.train_sizes_.tolist(),
            "train_mean": train_mean.tolist(),
            "train_std": train_std.tolist(),
            "val_mean": val_mean.tolist(),
            "val_std": val_std.tolist(),
            "gap": float(train_mean[-1] - val_mean[-1]),
        }
