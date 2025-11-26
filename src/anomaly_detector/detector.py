"""
Anomaly detection module.

This module implements hybrid anomaly detection combining LSTM reconstruction
errors with decrease-rate based detection.
"""

from typing import Tuple, Dict
import numpy as np
import pandas as pd
import logging

from .config import DetectionConfig


logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Hybrid anomaly detector combining LSTM and decrease-rate methods."""
    
    def __init__(self, config: DetectionConfig):
        """
        Initialize AnomalyDetector.
        
        Args:
            config: Detection configuration object.
        """
        self.config = config
        
    def detect_lstm_anomalies(
        self,
        test_errors: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        Detect anomalies based on LSTM reconstruction errors.
        
        Args:
            test_errors: Reconstruction errors for test data.
            threshold: Anomaly detection threshold.
            
        Returns:
            np.ndarray: Boolean array indicating anomalies.
        """
        anomaly_flags = test_errors > threshold
        n_anomalies = anomaly_flags.sum()
        
        logger.info(
            f"LSTM detected {n_anomalies} anomalies "
            f"({n_anomalies/len(test_errors)*100:.1f}%) using threshold {threshold:.6f}"
        )
        
        return anomaly_flags
    
    def detect_decrease_anomalies(
        self,
        time_series: pd.Series,
        threshold: float = None
    ) -> pd.Series:
        """
        Detect anomalies based on significant cost decreases.
        
        Args:
            time_series: Time series of costs.
            threshold: Decrease threshold (negative value). If None, uses config value.
            
        Returns:
            pd.Series: Boolean series indicating decrease anomalies.
        """
        if threshold is None:
            threshold = self.config.decrease_threshold
        
        pct_change = time_series.pct_change()
        decrease_flags = pct_change < threshold
        n_anomalies = decrease_flags.sum()
        
        logger.info(
            f"Decrease-rate detected {n_anomalies} anomalies "
            f"({n_anomalies/len(time_series)*100:.1f}%) using threshold {threshold*100:.0f}%"
        )
        
        return decrease_flags
    
    def calculate_hybrid_score(
        self,
        lstm_flags: np.ndarray,
        decrease_flags: np.ndarray
    ) -> np.ndarray:
        """
        Calculate hybrid anomaly score combining both detection methods.
        
        The score is a weighted combination of LSTM and decrease-rate flags:
        - 0.0: No anomaly detected by either method
        - 0.5: Anomaly detected by one method
        - 1.0: Strong anomaly detected by both methods
        
        Args:
            lstm_flags: Boolean array of LSTM anomalies.
            decrease_flags: Boolean array of decrease anomalies.
            
        Returns:
            np.ndarray: Hybrid anomaly scores.
            
        Raises:
            ValueError: If flag arrays have different lengths.
        """
        if len(lstm_flags) != len(decrease_flags):
            raise ValueError(
                f"Flag arrays must have same length: "
                f"lstm={len(lstm_flags)}, decrease={len(decrease_flags)}"
            )
        
        score = (
            lstm_flags.astype(float) * self.config.lstm_weight +
            decrease_flags.astype(float) * self.config.decrease_weight
        )
        
        n_strong = (score == 1.0).sum()
        n_weak = (score == 0.5).sum()
        
        logger.info(
            f"Hybrid scores: {n_strong} strong anomalies (score=1.0), "
            f"{n_weak} weak anomalies (score=0.5)"
        )
        
        return score
    
    def create_anomaly_dataframe(
        self,
        time_series: pd.Series,
        lstm_flags: np.ndarray,
        decrease_flags: np.ndarray,
        test_errors: np.ndarray,
        pct_change: pd.Series
    ) -> pd.DataFrame:
        """
        Create a comprehensive DataFrame with anomaly detection results.
        
        Args:
            time_series: Original time series data.
            lstm_flags: LSTM anomaly flags.
            decrease_flags: Decrease anomaly flags.
            test_errors: LSTM reconstruction errors.
            pct_change: Percentage changes in time series.
            
        Returns:
            pd.DataFrame: DataFrame with anomaly detection results.
        """
        anomaly_score = self.calculate_hybrid_score(lstm_flags, decrease_flags)
        
        df = pd.DataFrame({
            'TotalCost': time_series.values,
            'lstm_flag': lstm_flags.astype(int),
            'decrease_flag': decrease_flags.astype(int),
            'anomaly_score': anomaly_score,
            'pct_change': pct_change.values * 100,  # Convert to percentage
            'mse': test_errors
        }, index=time_series.index)
        
        return df
    
    def get_strong_anomalies(
        self,
        anomaly_df: pd.DataFrame,
        score_threshold: float = 1.0
    ) -> pd.DataFrame:
        """
        Filter for strong anomalies (score >= threshold).
        
        Args:
            anomaly_df: DataFrame with anomaly scores.
            score_threshold: Minimum score to consider as strong anomaly.
            
        Returns:
            pd.DataFrame: Filtered DataFrame containing only strong anomalies.
        """
        strong_anomalies = anomaly_df[anomaly_df['anomaly_score'] >= score_threshold].copy()
        
        logger.info(f"Found {len(strong_anomalies)} strong anomalies (score >= {score_threshold})")
        
        return strong_anomalies
    
    def calculate_metrics(
        self,
        predicted_flags: np.ndarray,
        true_flags: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate detection performance metrics.
        
        Args:
            predicted_flags: Predicted anomaly flags.
            true_flags: Ground truth anomaly flags.
            
        Returns:
            Dict containing precision, recall, and f1_score.
        """
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Handle edge cases where all predictions or all true values are 0
        precision = precision_score(true_flags, predicted_flags, zero_division=0)
        recall = recall_score(true_flags, predicted_flags, zero_division=0)
        f1 = f1_score(true_flags, predicted_flags, zero_division=0)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        logger.info(
            f"Metrics: Precision={precision:.3f}, "
            f"Recall={recall:.3f}, F1={f1:.3f}"
        )
        
        return metrics
