"""
Unit tests for anomaly detection framework.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Test imports
from anomaly_detector.config import Config, ModelConfig, DetectionConfig
from anomaly_detector.data_preprocessing import DataPreprocessor, generate_mock_data
from anomaly_detector.model import LSTMAutoencoder
from anomaly_detector.detector import AnomalyDetector


class TestConfig:
    """Test configuration classes."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = Config()
        assert config.model.window_size == 7
        assert config.detection.split_ratio == 0.8
        assert config.data.customer_id == 900
    
    def test_custom_config(self):
        """Test custom configuration."""
        model_config = ModelConfig(window_size=14, lstm_units=64)
        assert model_config.window_size == 14
        assert model_config.lstm_units == 64


class TestDataPreprocessing:
    """Test data preprocessing functions."""
    
    def test_generate_mock_data(self):
        """Test mock data generation."""
        df = generate_mock_data(customer_id=100, n_days=30)
        assert len(df) == 30
        assert 'Date' in df.columns
        assert 'TotalCost' in df.columns
        assert 'CustomerID' in df.columns
        assert df['CustomerID'].iloc[0] == 100
    
    def test_data_preprocessor_init(self):
        """Test DataPreprocessor initialization."""
        from anomaly_detector.config import DataConfig
        config = DataConfig()
        preprocessor = DataPreprocessor(config)
        assert preprocessor.config == config
        assert preprocessor.scaler is not None
    
    def test_create_sequences(self):
        """Test sequence creation."""
        data = np.arange(10).reshape(-1, 1)
        sequences = DataPreprocessor.create_sequences(data, window_size=3)
        assert sequences.shape == (8, 3, 1)
        assert np.array_equal(sequences[0], [[0], [1], [2]])
    
    def test_create_sequences_too_short(self):
        """Test sequence creation with insufficient data."""
        data = np.arange(2).reshape(-1, 1)
        with pytest.raises(ValueError):
            DataPreprocessor.create_sequences(data, window_size=5)
    
    def test_normalize_data(self):
        """Test data normalization."""
        from anomaly_detector.config import DataConfig
        preprocessor = DataPreprocessor(DataConfig())
        data = np.array([0, 5, 10]).reshape(-1, 1)
        normalized = preprocessor.normalize_data(data)
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0


class TestLSTMAutoencoder:
    """Test LSTM Autoencoder model."""
    
    def test_model_init(self):
        """Test model initialization."""
        config = ModelConfig()
        model = LSTMAutoencoder(config)
        assert model.config == config
        assert model.model is None
    
    def test_build_model(self):
        """Test model building."""
        config = ModelConfig(window_size=7, lstm_units=16)
        model = LSTMAutoencoder(config)
        keras_model = model.build_model()
        assert keras_model is not None
        assert len(keras_model.layers) > 0
    
    def test_calculate_threshold(self):
        """Test threshold calculation."""
        config = ModelConfig()
        model = LSTMAutoencoder(config)
        train_errors = np.array([0.1, 0.2, 0.15, 0.18, 0.12])
        threshold = model.calculate_threshold(train_errors, sigma_multiplier=1.5)
        expected = train_errors.mean() + 1.5 * train_errors.std()
        assert abs(threshold - expected) < 1e-6


class TestAnomalyDetector:
    """Test anomaly detector."""
    
    def test_detector_init(self):
        """Test detector initialization."""
        config = DetectionConfig()
        detector = AnomalyDetector(config)
        assert detector.config == config
    
    def test_detect_lstm_anomalies(self):
        """Test LSTM anomaly detection."""
        config = DetectionConfig()
        detector = AnomalyDetector(config)
        test_errors = np.array([0.1, 0.5, 0.15, 0.6, 0.12])
        threshold = 0.3
        anomalies = detector.detect_lstm_anomalies(test_errors, threshold)
        expected = np.array([False, True, False, True, False])
        assert np.array_equal(anomalies, expected)
    
    def test_calculate_hybrid_score(self):
        """Test hybrid score calculation."""
        config = DetectionConfig(lstm_weight=0.5, decrease_weight=0.5)
        detector = AnomalyDetector(config)
        lstm_flags = np.array([True, False, True])
        decrease_flags = np.array([True, True, False])
        scores = detector.calculate_hybrid_score(lstm_flags, decrease_flags)
        expected = np.array([1.0, 0.5, 0.5])
        assert np.allclose(scores, expected)
    
    def test_hybrid_score_length_mismatch(self):
        """Test hybrid score with mismatched lengths."""
        config = DetectionConfig()
        detector = AnomalyDetector(config)
        lstm_flags = np.array([True, False])
        decrease_flags = np.array([True])
        with pytest.raises(ValueError):
            detector.calculate_hybrid_score(lstm_flags, decrease_flags)


class TestIntegration:
    """Integration tests for the pipeline."""
    
    def test_pipeline_with_mock_data(self):
        """Test complete pipeline with mock data."""
        from anomaly_detector.pipeline import AnomalyDetectionPipeline
        
        config = Config()
        config.model.epochs = 2  # Fast test
        config.model.window_size = 7
        
        pipeline = AnomalyDetectionPipeline(config)
        
        # This should run without errors
        try:
            anomaly_df, explained_anomalies = pipeline.run(
                use_mock_data=True,
                use_llm_api=False,
                verbose=0
            )
            assert anomaly_df is not None
            assert len(anomaly_df) > 0
            assert 'anomaly_score' in anomaly_df.columns
        except Exception as e:
            pytest.fail(f"Pipeline failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
