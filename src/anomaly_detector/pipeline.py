"""
Main pipeline for cloud usage anomaly detection.

This module orchestrates the entire anomaly detection workflow:
1. Data loading and preprocessing
2. LSTM Autoencoder model training
3. Anomaly detection using hybrid approach
4. LLM-based explanation generation
5. Visualization of results
"""

from typing import Optional, Tuple
import logging
import pandas as pd
import numpy as np

from .config import Config
from .data_preprocessing import DataPreprocessor, generate_mock_data
from .model import LSTMAutoencoder
from .detector import AnomalyDetector
from .explainer import ExplanationGenerator
from .visualization import AnomalyVisualizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnomalyDetectionPipeline:
    """Main pipeline for anomaly detection."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the anomaly detection pipeline.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or Config()
        
        # Initialize components
        self.data_preprocessor = DataPreprocessor(self.config.data)
        self.model = LSTMAutoencoder(self.config.model)
        self.detector = AnomalyDetector(self.config.detection)
        self.explainer = ExplanationGenerator(self.config.llm)
        self.visualizer = AnomalyVisualizer()
        
        # State variables
        self.time_series: Optional[pd.Series] = None
        self.anomaly_df: Optional[pd.DataFrame] = None
        self.strong_anomalies: Optional[pd.DataFrame] = None
        
        logger.info("Anomaly detection pipeline initialized")
    
    def load_data(
        self,
        df: Optional[pd.DataFrame] = None,
        use_mock_data: bool = False
    ) -> pd.Series:
        """
        Load and prepare time series data.
        
        Args:
            df: Input DataFrame. If None and use_mock_data=False, raises error.
            use_mock_data: Whether to generate mock data for testing.
            
        Returns:
            pd.Series: Prepared time series data.
            
        Raises:
            ValueError: If no data is provided and use_mock_data is False.
        """
        if use_mock_data:
            logger.info("Generating mock data for testing")
            df = generate_mock_data(
                customer_id=self.config.data.customer_id,
                random_seed=self.config.data.random_seed
            )
        elif df is None:
            raise ValueError("Must provide data DataFrame or set use_mock_data=True")
        
        self.time_series = self.data_preprocessor.load_and_prepare_data(df)
        
        logger.info(
            f"Loaded time series: {len(self.time_series)} data points "
            f"for customer {self.config.data.customer_id}"
        )
        
        return self.time_series
    
    def prepare_sequences(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training and testing sequences.
        
        Returns:
            Tuple of (X_train, Y_train, X_test, Y_test).
            
        Raises:
            RuntimeError: If data has not been loaded.
        """
        if self.time_series is None:
            raise RuntimeError("Data must be loaded first. Call load_data().")
        
        # Normalize data
        ts_scaled = self.data_preprocessor.normalize_data(self.time_series.values)
        
        # Create sequences
        sequences = self.data_preprocessor.create_sequences(
            ts_scaled,
            self.config.model.window_size
        )
        
        # Split into train/test
        X_train, Y_train, X_test, Y_test = \
            self.data_preprocessor.prepare_train_test_split(
                sequences,
                self.config.detection.split_ratio
            )
        
        logger.info(
            f"Prepared sequences: {len(X_train)} train, {len(X_test)} test"
        )
        
        return X_train, Y_train, X_test, Y_test
    
    def train_model(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        verbose: int = 0
    ):
        """
        Train the LSTM Autoencoder model.
        
        Args:
            X_train: Training input sequences.
            Y_train: Training target sequences.
            verbose: Verbosity level for training.
        """
        logger.info("Building and training LSTM Autoencoder model")
        
        self.model.build_model()
        self.model.train(X_train, Y_train, verbose=verbose)
        
        logger.info("Model training completed")
    
    def detect_anomalies(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Detect anomalies using hybrid approach.
        
        Args:
            X_train: Training sequences (for threshold calculation).
            X_test: Test sequences.
            
        Returns:
            pd.DataFrame: DataFrame with anomaly detection results.
            
        Raises:
            RuntimeError: If time series data is not loaded.
        """
        if self.time_series is None:
            raise RuntimeError("Time series data not loaded")
        
        logger.info("Starting anomaly detection")
        
        # Calculate reconstruction errors
        train_errors = self.model.calculate_reconstruction_error(X_train)
        test_errors = self.model.calculate_reconstruction_error(X_test)
        
        # Calculate threshold
        threshold = self.model.calculate_threshold(
            train_errors,
            self.config.detection.lstm_sigma_multiplier
        )
        
        # Detect LSTM anomalies
        lstm_flags = self.detector.detect_lstm_anomalies(test_errors, threshold)
        
        # Get test period time series
        split_point = int(
            (len(self.time_series) - self.config.model.window_size + 1) *
            self.config.detection.split_ratio
        )
        test_start_idx = self.config.model.window_size + split_point
        ts_test = self.time_series[test_start_idx:test_start_idx + len(test_errors)]
        
        # Detect decrease anomalies
        pct_change = ts_test.pct_change()
        decrease_flags = self.detector.detect_decrease_anomalies(ts_test).values
        
        # Create anomaly DataFrame
        self.anomaly_df = self.detector.create_anomaly_dataframe(
            ts_test,
            lstm_flags,
            decrease_flags,
            test_errors,
            pct_change
        )
        
        # Get strong anomalies
        self.strong_anomalies = self.detector.get_strong_anomalies(self.anomaly_df)
        
        logger.info(
            f"Anomaly detection completed: "
            f"{len(self.strong_anomalies)} strong anomalies found"
        )
        
        return self.anomaly_df
    
    def generate_explanations(self, use_api: bool = False) -> pd.DataFrame:
        """
        Generate explanations for strong anomalies.
        
        Args:
            use_api: Whether to use LLM API for explanations.
            
        Returns:
            pd.DataFrame: Strong anomalies with explanations.
            
        Raises:
            RuntimeError: If anomalies have not been detected.
        """
        if self.strong_anomalies is None:
            raise RuntimeError(
                "Anomalies must be detected first. Call detect_anomalies()."
            )
        
        if self.strong_anomalies.empty:
            logger.info("No strong anomalies to explain")
            return pd.DataFrame()
        
        logger.info(
            f"Generating explanations for {len(self.strong_anomalies)} strong anomalies"
        )
        
        explained_anomalies = self.explainer.generate_batch_explanations(
            self.strong_anomalies,
            use_api=use_api
        )
        
        return explained_anomalies
    
    def visualize_results(self, save_path: Optional[str] = None):
        """
        Visualize anomaly detection results.
        
        Args:
            save_path: Optional path to save the figure.
            
        Returns:
            plt.Figure: The created figure.
            
        Raises:
            RuntimeError: If anomalies have not been detected.
        """
        if self.anomaly_df is None or self.strong_anomalies is None:
            raise RuntimeError(
                "Anomalies must be detected first. Call detect_anomalies()."
            )
        
        logger.info("Generating visualization")
        
        fig = self.visualizer.plot_anomaly_results(
            self.anomaly_df,
            self.strong_anomalies,
            self.config.data.customer_id,
            save_path=save_path
        )
        
        return fig
    
    def run(
        self,
        df: Optional[pd.DataFrame] = None,
        use_mock_data: bool = True,
        use_llm_api: bool = False,
        verbose: int = 0,
        save_plot: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete anomaly detection pipeline.
        
        Args:
            df: Input DataFrame. If None, uses mock data.
            use_mock_data: Whether to generate mock data (if df is None).
            use_llm_api: Whether to use LLM API for explanations.
            verbose: Verbosity level for training.
            save_plot: Optional path to save visualization.
            
        Returns:
            Tuple of (anomaly_df, explained_anomalies).
        """
        logger.info("=" * 60)
        logger.info("Starting Anomaly Detection Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load data
        self.load_data(df=df, use_mock_data=use_mock_data)
        
        # Step 2: Prepare sequences
        X_train, Y_train, X_test, Y_test = self.prepare_sequences()
        
        # Step 3: Train model
        self.train_model(X_train, Y_train, verbose=verbose)
        
        # Step 4: Detect anomalies
        self.detect_anomalies(X_train, X_test)
        
        # Step 5: Generate explanations
        explained_anomalies = self.generate_explanations(use_api=use_llm_api)
        
        # Step 6: Visualize results
        self.visualize_results(save_path=save_plot)
        
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully")
        logger.info("=" * 60)
        
        return self.anomaly_df, explained_anomalies
