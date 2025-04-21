from wiz.shared.estimator import estimator
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from typing import Optional, Union


class PyTorchLinearRegression(estimator.Regressor):
    def __init__(self, lr: float = 0.01, epochs: int = 100, normalize: bool = False):
        self.lr = lr
        self.epochs = epochs
        self.normalize = normalize
        self.model: Optional[nn.Linear] = None
        self.scaler: Optional[StandardScaler] = StandardScaler() if normalize else None
        self.feature_names: list[str] = []

    def fit(
        self, X: Union[np.ndarray, pl.DataFrame], y: Union[np.ndarray, pl.Series]
    ) -> "PyTorchLinearRegression":
        X, y = self._check_input(X, y)

        if isinstance(X, pl.DataFrame):
            self.feature_names = X.columns  # Store feature names

        if self.normalize:
            X = self.scaler.fit_transform(X)

        n_features: int = X.shape[1]
        self.model = nn.Linear(n_features, 1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

        X_tensor: torch.Tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor: torch.Tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        for epoch in range(self.epochs):
            self.model.train()
            y_pred = self.model(X_tensor)
            loss = self.criterion(y_pred, y_tensor)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return self

    def _predict(self, X: Union[np.ndarray, pl.DataFrame]) -> np.ndarray:
        X = self._check_input(X)
        if self.normalize:
            X = self.scaler.transform(X)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model(X_tensor).numpy()

    def score(
        self, X: Union[np.ndarray, pl.DataFrame], y: Union[np.ndarray, pl.Series]
    ) -> float:
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def feature_importance(self) -> pl.DataFrame:
        """Returns feature importance as absolute weight values."""
        if self.model is None:
            raise ValueError("Model is not trained yet. Call `fit` first.")

        weights = self.model.weight.detach().numpy().flatten()
        importance = np.abs(weights)  # Absolute magnitude of weights

        feature_names = (
            self.feature_names
            if self.feature_names
            else [f"Feature {i}" for i in range(len(weights))]
        )

        return pl.DataFrame({"Feature": feature_names, "Importance": importance}).sort(
            "Importance", descending=True
        )

    def _check_input(
        self,
        X: Union[np.ndarray, pl.DataFrame],
        y: Optional[Union[np.ndarray, pl.Series]] = None,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Converts Polars DataFrame/Series to NumPy and ensures correct shape."""
        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()
        if y is not None:
            if isinstance(y, pl.Series):
                y = y.to_numpy()
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        return np.array(X, dtype=np.float32)


# Generate synthetic multivariate data using Polars
np.random.seed(42)
X_np: np.ndarray = np.random.rand(100, 2) * 10  # Two features
true_weights: np.ndarray = np.array([2.5, -1.2])
y_np: np.ndarray = (
    X_np @ true_weights + np.random.randn(100) * 2
)  # Linear relationship with noise

# Convert to Polars DataFrame
X_pl: pl.DataFrame = pl.DataFrame(X_np, schema=["Feature 1", "Feature 2"])
y_pl: pl.Series = pl.Series("Target", y_np)

# Train the model
model = PyTorchLinearRegression(lr=0.01, epochs=200, normalize=True)
model.fit(X_pl, y_pl)

# Make predictions
y_pred: np.ndarray = model.predict(X_pl)

# Evaluate the model
r2: float = model.score(X_pl, y_pl)
print(f"R² Score: {r2:.4f}")

# Get feature importance
importance_df = model.feature_importance()
print("\nFeature Importance:\n", importance_df)

from sklearn import linear_model

m = linear_model.LinearRegression()

m.fit(X_pl, y_pl)
y_pred2 = m.predict(X_pl)

# Plot feature importance
plt.bar(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Features")
plt.ylabel("Importance (Absolute Weight)")
plt.title("Feature Importance in Linear Regression")
plt.show()

# Plot results (only for single-feature case)
if X_np.shape[1] == 1:
    plt.scatter(X_np, y_np, label="Original data", alpha=0.7)
    plt.plot(X_np, y_pred, "r-", label="Fitted line", linewidth=2)
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression with PyTorch (Polars API)")
    plt.show()

print(1)
