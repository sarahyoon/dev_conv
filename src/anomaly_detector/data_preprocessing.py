"""
Data preprocessing module for anomaly detection.

This module handles data loading, cleaning, and sequence generation
for the LSTM autoencoder model.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

from .config import DataConfig


logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing for anomaly detection."""
    
    def __init__(self, config: DataConfig):
        """
        Initialize DataPreprocessor.
        
        Args:
            config: Data configuration object.
        """
        self.config = config
        self.scaler = MinMaxScaler()
        
    def load_and_prepare_data(
        self, 
        df: pd.DataFrame
    ) -> pd.Series:
        """
        Load and prepare time series data for a specific customer.
        
        Args:
            df: Input DataFrame containing customer data.
            
        Returns:
            pd.Series: Time series of total costs indexed by date.
            
        Raises:
            ValueError: If required columns are missing or no data for customer.
        """
        required_cols = [
            'CustomerID', 
            self.config.date_column, 
            self.config.cost_column
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter for specific customer
        customer_data = df[df['CustomerID'] == self.config.customer_id]
        if customer_data.empty:
            raise ValueError(f"No data found for customer ID {self.config.customer_id}")
        
        # Ensure date is datetime type
        customer_data = customer_data.copy()
        customer_data[self.config.date_column] = pd.to_datetime(
            customer_data[self.config.date_column]
        )
        
        # Create time series
        ts = customer_data.set_index(
            self.config.date_column
        )[self.config.cost_column].sort_index()
        
        logger.info(
            f"Loaded time series for customer {self.config.customer_id}: "
            f"{len(ts)} data points from {ts.index.min()} to {ts.index.max()}"
        )
        
        return ts
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data using MinMaxScaler.
        
        Args:
            data: Input data array.
            
        Returns:
            np.ndarray: Normalized data.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return self.scaler.fit_transform(data)
    
    def denormalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Denormalize data back to original scale.
        
        Args:
            data: Normalized data array.
            
        Returns:
            np.ndarray: Denormalized data.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return self.scaler.inverse_transform(data)
    
    @staticmethod
    def create_sequences(
        data: np.ndarray, 
        window_size: int
    ) -> np.ndarray:
        """
        Create sliding window sequences from time series data.
        
        Args:
            data: Input time series data.
            window_size: Size of the sliding window.
            
        Returns:
            np.ndarray: Array of sequences with shape (n_sequences, window_size, n_features).
            
        Raises:
            ValueError: If data is too short for the window size.
        """
        if len(data) < window_size:
            raise ValueError(
                f"Data length ({len(data)}) must be >= window_size ({window_size})"
            )
        
        sequences = []
        for i in range(len(data) - window_size + 1):
            sequences.append(data[i:i + window_size])
        
        sequences_array = np.array(sequences)
        logger.info(
            f"Created {len(sequences_array)} sequences with window size {window_size}"
        )
        
        return sequences_array
    
    def prepare_train_test_split(
        self,
        sequences: np.ndarray,
        split_ratio: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split sequences into training and testing sets.
        
        Args:
            sequences: Array of sequences.
            split_ratio: Ratio of training data (0-1).
            
        Returns:
            Tuple of (X_train, Y_train, X_test, Y_test).
            For autoencoder, X and Y are the same.
            
        Raises:
            ValueError: If split_ratio is not between 0 and 1.
        """
        if not 0 < split_ratio < 1:
            raise ValueError(f"split_ratio must be between 0 and 1, got {split_ratio}")
        
        split_point = int(len(sequences) * split_ratio)
        
        X_train = sequences[:split_point]
        X_test = sequences[split_point:]
        
        # For autoencoder, input equals output
        Y_train = X_train
        Y_test = X_test
        
        logger.info(
            f"Split data: {len(X_train)} training sequences, "
            f"{len(X_test)} test sequences"
        )
        
        return X_train, Y_train, X_test, Y_test


def generate_mock_data(
    customer_id: int = 900,
    n_days: int = 180,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate mock time series data for testing.
    
    This function creates synthetic cloud usage data with normal patterns
    and injected anomalies for demonstration purposes.
    
    Args:
        customer_id: Customer ID for the mock data.
        n_days: Number of days to generate.
        random_seed: Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: Mock data with columns matching the expected schema.
    """
    np.random.seed(random_seed)
    
    date_range = pd.date_range(start='2025-01-01', periods=n_days)
    
    # Generate normal pattern with seasonality and noise
    normal_pattern = np.sin(np.arange(n_days) / 7 * 2 * np.pi) * 100 + 500
    noise = np.random.normal(0, 50, n_days)
    total_cost = (normal_pattern + noise).clip(min=100)
    
    # Inject anomaly (sharp cost decrease)
    anomaly_start_idx = int(n_days * 0.83)  # Around 150 for 180 days
    if anomaly_start_idx < n_days:
        total_cost[anomaly_start_idx:anomaly_start_idx + 3] = (
            total_cost[anomaly_start_idx - 1] * 0.4
        )
    
    # Create mock dataframe
    mock_data = {
        'Date': date_range,
        'TotalCost': total_cost,
        'CustomerID': customer_id,
        'MaskedSub': np.random.choice(['sub_A123', 'sub_B456', 'sub_C789'], size=n_days),
        'MeterCategory': np.random.choice(['Virtual Machines', 'Storage', 'Database'], size=n_days),
        'MeterSubCategory': np.random.choice(['Esv5', 'Premium SSD', 'PostgreSQL'], size=n_days)
    }
    
    df = pd.DataFrame(mock_data)
    df['Date'] = pd.to_datetime(df['Date'])
    
    logger.info(f"Generated mock data: {len(df)} rows for customer {customer_id}")
    
    return df
