"""
LSTM Autoencoder model module.

This module defines and trains the LSTM autoencoder for anomaly detection.
"""

from typing import Tuple, Optional
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, History
import logging

from .config import ModelConfig


logger = logging.getLogger(__name__)


class LSTMAutoencoder:
    """LSTM Autoencoder for time series anomaly detection."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize LSTM Autoencoder.
        
        Args:
            config: Model configuration object.
        """
        self.config = config
        self.model: Optional[Model] = None
        self.history: Optional[History] = None
        
    def build_model(self) -> Model:
        """
        Build the LSTM Autoencoder architecture.
        
        Returns:
            Model: Compiled Keras model.
        """
        model = Sequential([
            # Encoder: Compress the sequence into a fixed-size vector
            LSTM(
                self.config.lstm_units,
                activation=self.config.activation,
                input_shape=(self.config.window_size, 1),
                return_sequences=False,
                name='encoder_lstm'
            ),
            
            # Repeat the encoded vector for decoding
            RepeatVector(self.config.window_size, name='repeat_vector'),
            
            # Decoder: Reconstruct the original sequence
            LSTM(
                self.config.lstm_units,
                activation=self.config.activation,
                return_sequences=True,
                name='decoder_lstm'
            ),
            
            # Output layer: Reconstruct the original values
            TimeDistributed(Dense(1), name='output')
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        logger.info(f"Built LSTM Autoencoder model with {self.config.lstm_units} units")
        logger.debug(f"Model summary:\n{model.summary()}")
        
        self.model = model
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        verbose: int = 0
    ) -> History:
        """
        Train the LSTM Autoencoder model.
        
        Args:
            X_train: Training input sequences.
            Y_train: Training target sequences (same as X for autoencoder).
            verbose: Verbosity level for training (0=silent, 1=progress bar, 2=one line per epoch).
            
        Returns:
            History: Training history object.
            
        Raises:
            RuntimeError: If model has not been built.
        """
        if self.model is None:
            raise RuntimeError("Model must be built before training. Call build_model() first.")
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.config.patience,
            restore_best_weights=True,
            verbose=verbose
        )
        
        logger.info(
            f"Starting training: epochs={self.config.epochs}, "
            f"batch_size={self.config.batch_size}, "
            f"validation_split={self.config.validation_split}"
        )
        
        self.history = self.model.fit(
            X_train, Y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            shuffle=False,  # Don't shuffle to maintain temporal order
            callbacks=[early_stop],
            verbose=verbose
        )
        
        logger.info("Training completed successfully")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            X: Input sequences.
            
        Returns:
            np.ndarray: Reconstructed sequences.
            
        Raises:
            RuntimeError: If model has not been trained.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction. Call train() first.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def calculate_reconstruction_error(
        self,
        X: np.ndarray,
        Y_pred: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate Mean Squared Error (MSE) for reconstruction.
        
        Args:
            X: Original sequences.
            Y_pred: Predicted sequences. If None, will call predict(X).
            
        Returns:
            np.ndarray: MSE for each sequence.
        """
        if Y_pred is None:
            Y_pred = self.predict(X)
        
        # Calculate MSE across time steps and features
        mse = np.mean(np.square(Y_pred - X), axis=(1, 2))
        
        logger.debug(f"Calculated reconstruction error: mean={mse.mean():.6f}, std={mse.std():.6f}")
        
        return mse
    
    def calculate_threshold(
        self,
        train_errors: np.ndarray,
        sigma_multiplier: float = 1.5
    ) -> float:
        """
        Calculate anomaly detection threshold based on training errors.
        
        Uses mean + sigma_multiplier * std as the threshold.
        
        Args:
            train_errors: Reconstruction errors from training data.
            sigma_multiplier: Number of standard deviations above mean.
            
        Returns:
            float: Anomaly detection threshold.
        """
        threshold = train_errors.mean() + sigma_multiplier * train_errors.std()
        
        logger.info(
            f"Calculated threshold: {threshold:.6f} "
            f"(mean={train_errors.mean():.6f}, "
            f"std={train_errors.std():.6f}, "
            f"multiplier={sigma_multiplier})"
        )
        
        return threshold
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where model should be saved.
            
        Raises:
            RuntimeError: If model has not been trained.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train the model first.")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model.
        """
        from tensorflow.keras.models import load_model
        
        self.model = load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
