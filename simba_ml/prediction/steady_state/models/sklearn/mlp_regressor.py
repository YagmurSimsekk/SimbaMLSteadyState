"""Multi-layer Perceptron Regressor for steady-state prediction using sklearn."""
import dataclasses
from typing import Optional
import numpy as np
import numpy.typing as npt
from sklearn.neural_network import MLPRegressor as SklearnMLP
from sklearn.preprocessing import StandardScaler


@dataclasses.dataclass
class MLPConfig:
    """Configuration for MLP Regressor."""

    hidden_layer_sizes: tuple = (64, 32)
    activation: str = 'relu'
    solver: str = 'adam'
    alpha: float = 0.0001  # L2 regularization
    batch_size: int = 'auto'
    learning_rate_init: float = 0.001
    max_iter: int = 1000
    early_stopping: bool = True
    validation_fraction: float = 0.2
    n_iter_no_change: int = 30
    random_state: int = 42
    verbose: bool = False


class MLPRegressor:
    """Multi-layer Perceptron for steady-state regression.

    Simple sklearn-based neural network for regression tasks.
    Auto-detects input/output dimensions from data.

    Args:
        config: Model configuration (optional)

    Example:
        >>> model = MLPRegressor()
        >>> model.train(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, config: Optional[MLPConfig] = None):
        """Initialize the MLP regressor.

        Args:
            config: Model configuration (uses defaults if None)
        """
        self.config = config or MLPConfig()
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def train(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64]
    ) -> None:
        """Train the model.

        Args:
            X_train: Training input features (n_samples, n_inputs)
            y_train: Training output features (n_samples, n_outputs)
        """
        # Normalize data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)

        # Create and train model
        self.model = SklearnMLP(
            hidden_layer_sizes=self.config.hidden_layer_sizes,
            activation=self.config.activation,
            solver=self.config.solver,
            alpha=self.config.alpha,
            batch_size=self.config.batch_size,
            learning_rate_init=self.config.learning_rate_init,
            max_iter=self.config.max_iter,
            early_stopping=self.config.early_stopping,
            validation_fraction=self.config.validation_fraction,
            n_iter_no_change=self.config.n_iter_no_change,
            random_state=self.config.random_state,
            verbose=self.config.verbose
        )

        self.model.fit(X_train_scaled, y_train_scaled)

    def predict(
        self,
        X: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Make predictions.

        Args:
            X: Input features (n_samples, n_inputs)

        Returns:
            Predictions (n_samples, n_outputs)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        return y_pred

    def save(self, filepath: str) -> None:
        """Save model to disk.

        Args:
            filepath: Path to save the model
        """
        import pickle

        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'config': self.config
            }, f)

    def load(self, filepath: str) -> None:
        """Load model from disk.

        Args:
            filepath: Path to load the model from
        """
        import pickle

        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler_X = data['scaler_X']
            self.scaler_y = data['scaler_y']
            self.config = data['config']
