"""Dense neural network for steady-state regression."""
import dataclasses
from typing import Optional

import numpy as np
import numpy.typing as npt

try:
    import tensorflow as tf
except ImportError as e:
    raise ImportError(
        "TensorFlow is not installed. Please install it to use Keras models."
    ) from e

if tuple(int(v) for v in tf.version.VERSION.split(".")) < (2, 10, 0):
    raise ImportError(
        "TensorFlow version 2.10.0 or higher is required."
    )


@dataclasses.dataclass
class ArchitectureParams:
    """Defines the parameters for the neural network architecture."""

    units: list[int] = dataclasses.field(default_factory=lambda: [64, 32])
    activation: str = "relu"
    dropout_rate: float = 0.0  # 0.0 = no dropout


@dataclasses.dataclass
class TrainingParams:
    """Defines the parameters for model training."""

    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    patience: int = 10  # Early stopping patience
    verbose: int = 0


@dataclasses.dataclass
class DenseNeuralNetworkConfig:
    """Configuration for Dense Neural Network steady-state predictor."""

    architecture_params: ArchitectureParams = dataclasses.field(
        default_factory=ArchitectureParams
    )
    training_params: TrainingParams = dataclasses.field(
        default_factory=TrainingParams
    )
    normalize: bool = True
    seed: int = 42
    name: str = "Keras Dense Neural Network (Steady-State)"


class DenseNeuralNetwork:
    """Dense neural network for steady-state prediction.

    Simple feedforward network for regression: inputs → hidden layers → outputs.
    No temporal dimension - suitable for steady-state mapping.

    Input/output dimensions are automatically detected from training data.

    Args:
        config: Model configuration (optional, uses defaults if None)

    Example:
        >>> # Simple usage - auto-detects dimensions from data
        >>> model = DenseNeuralNetwork()
        >>> model.train(X_train, y_train)  # Shapes: (1000, 43), (1000, 16)
        >>> predictions = model.predict(X_test)
        >>>
        >>> # With custom architecture
        >>> config = DenseNeuralNetworkConfig(
        ...     architecture_params=ArchitectureParams(units=[128, 64, 32]),
        ...     training_params=TrainingParams(epochs=100, batch_size=32)
        ... )
        >>> model = DenseNeuralNetwork(config=config)
        >>> model.train(X_train, y_train)
    """

    def __init__(
        self,
        config: Optional[DenseNeuralNetworkConfig] = None
    ):
        """Initialize the dense neural network.

        Args:
            config: Model configuration (uses defaults if None)
        """
        self.config = config or DenseNeuralNetworkConfig()

        # Dimensions auto-detected during training
        self.n_inputs = None
        self.n_outputs = None

        # Set random seed for reproducibility
        tf.random.set_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Model built during training after dimensions are known
        self.model = None

        # Normalization stats (computed during training)
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

        # Training history
        self.history = None

    def _build_model(self) -> tf.keras.Model:
        """Build the neural network architecture.

        Returns:
            Compiled Keras model
        """
        layers = []

        # Input layer
        layers.append(tf.keras.layers.InputLayer(input_shape=(self.n_inputs,)))

        # Hidden layers
        for units in self.config.architecture_params.units:
            layers.append(
                tf.keras.layers.Dense(
                    units=units,
                    activation=self.config.architecture_params.activation
                )
            )
            # Add dropout if specified
            if self.config.architecture_params.dropout_rate > 0:
                layers.append(
                    tf.keras.layers.Dropout(self.config.architecture_params.dropout_rate)
                )

        # Output layer (linear activation for regression)
        layers.append(tf.keras.layers.Dense(units=self.n_outputs))

        # Create sequential model
        model = tf.keras.Sequential(layers)

        # Compile model
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        return model

    def _normalize_data(
        self,
        X: npt.NDArray[np.float64],
        y: Optional[npt.NDArray[np.float64]] = None,
        fit: bool = False
    ) -> tuple:
        """Normalize input and output data.

        Args:
            X: Input features
            y: Output features (optional)
            fit: Whether to compute normalization stats

        Returns:
            Tuple of (X_normalized, y_normalized) or just X_normalized if y is None
        """
        if fit:
            # Compute normalization stats
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0) + 1e-8  # Avoid division by zero

            if y is not None:
                self.y_mean = np.mean(y, axis=0)
                self.y_std = np.std(y, axis=0) + 1e-8

        # Normalize X
        X_normalized = (X - self.X_mean) / self.X_std

        # Normalize y if provided
        if y is not None:
            y_normalized = (y - self.y_mean) / self.y_std
            return X_normalized, y_normalized
        else:
            return X_normalized

    def _denormalize_predictions(
        self,
        y_normalized: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Denormalize model predictions.

        Args:
            y_normalized: Normalized predictions

        Returns:
            Original scale predictions
        """
        return y_normalized * self.y_std + self.y_mean

    def train(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64]
    ) -> None:
        """Train the model.

        Automatically detects input/output dimensions from data shapes.

        Args:
            X_train: Training input features (n_samples, n_inputs)
            y_train: Training output features (n_samples, n_outputs)
        """
        # Auto-detect dimensions from data
        if self.n_inputs is None or self.n_outputs is None:
            self.n_inputs = X_train.shape[1]
            self.n_outputs = y_train.shape[1] if y_train.ndim > 1 else 1

            # Build model now that dimensions are known
            self.model = self._build_model()

        # Normalize data if configured
        if self.config.normalize:
            X_train_norm, y_train_norm = self._normalize_data(
                X_train, y_train, fit=True
            )
        else:
            X_train_norm, y_train_norm = X_train, y_train

        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.training_params.patience,
            mode='min',
            restore_best_weights=True
        )

        # Train model
        self.history = self.model.fit(
            X_train_norm,
            y_train_norm,
            epochs=self.config.training_params.epochs,
            batch_size=self.config.training_params.batch_size,
            validation_split=self.config.training_params.validation_split,
            callbacks=[early_stopping],
            verbose=self.config.training_params.verbose
        )

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
        # Normalize input if configured
        if self.config.normalize:
            X_norm = self._normalize_data(X, fit=False)
        else:
            X_norm = X

        # Predict
        y_pred_norm = self.model.predict(
            X_norm,
            verbose=self.config.training_params.verbose
        )

        # Denormalize predictions if configured
        if self.config.normalize:
            y_pred = self._denormalize_predictions(y_pred_norm)
        else:
            y_pred = y_pred_norm

        return y_pred

    def save(self, filepath: str) -> None:
        """Save model to disk.

        Args:
            filepath: Path to save the model (without extension)
        """
        # Save Keras model
        self.model.save(f"{filepath}.keras")

        # Save normalization stats
        if self.config.normalize:
            np.savez(
                f"{filepath}_norm_stats.npz",
                X_mean=self.X_mean,
                X_std=self.X_std,
                y_mean=self.y_mean,
                y_std=self.y_std
            )

    def load(self, filepath: str) -> None:
        """Load model from disk.

        Args:
            filepath: Path to load the model from (without extension)
        """
        # Load Keras model
        self.model = tf.keras.models.load_model(f"{filepath}.keras")

        # Load normalization stats
        if self.config.normalize:
            stats = np.load(f"{filepath}_norm_stats.npz")
            self.X_mean = stats['X_mean']
            self.X_std = stats['X_std']
            self.y_mean = stats['y_mean']
            self.y_std = stats['y_std']
