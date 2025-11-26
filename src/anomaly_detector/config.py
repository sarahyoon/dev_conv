"""
Configuration module for the anomaly detection framework.

This module contains all configuration parameters used throughout the system.
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class ModelConfig:
    """Configuration for LSTM Autoencoder model."""
    
    window_size: int = 7
    """Number of days in the input sequence (7 days)."""
    
    lstm_units: int = 32
    """Number of LSTM units in encoder and decoder."""
    
    epochs: int = 50
    """Maximum number of training epochs."""
    
    batch_size: int = 16
    """Batch size for training."""
    
    validation_split: float = 0.1
    """Fraction of training data to use for validation."""
    
    patience: int = 5
    """Number of epochs with no improvement before early stopping."""
    
    activation: str = 'relu'
    """Activation function for LSTM layers."""


@dataclass
class DetectionConfig:
    """Configuration for anomaly detection."""
    
    split_ratio: float = 0.8
    """Train/test split ratio."""
    
    decrease_threshold: float = -0.3
    """Threshold for detecting significant cost decrease (-30%)."""
    
    lstm_sigma_multiplier: float = 1.5
    """Multiplier for standard deviation in LSTM anomaly threshold."""
    
    lstm_weight: float = 0.5
    """Weight for LSTM flag in hybrid anomaly score."""
    
    decrease_weight: float = 0.5
    """Weight for decrease flag in hybrid anomaly score."""


@dataclass
class LLMConfig:
    """Configuration for LLM-based explanation generation."""
    
    api_key: Optional[str] = None
    """API key for Gemini LLM service."""
    
    api_url_template: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    """URL template for Gemini API."""
    
    max_explanation_length: int = 70
    """Maximum length for explanation text (in characters)."""
    
    def __post_init__(self):
        """Initialize API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.environ.get('GEMINI_API_KEY', '')
    
    @property
    def api_url(self) -> str:
        """Get the complete API URL with key."""
        return self.api_url_template.format(api_key=self.api_key)


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    customer_id: int = 900
    """Target customer ID for analysis."""
    
    date_column: str = 'Date'
    """Name of the date column in dataset."""
    
    cost_column: str = 'TotalCost'
    """Name of the cost column in dataset."""
    
    random_seed: int = 42
    """Random seed for reproducibility."""


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""
    
    model: ModelConfig = None
    detection: DetectionConfig = None
    llm: LLMConfig = None
    data: DataConfig = None
    
    def __post_init__(self):
        """Initialize sub-configurations if not provided."""
        if self.model is None:
            self.model = ModelConfig()
        if self.detection is None:
            self.detection = DetectionConfig()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.data is None:
            self.data = DataConfig()
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """
        Create Config from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values.
            
        Returns:
            Config: Configured Config object.
        """
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            detection=DetectionConfig(**config_dict.get('detection', {})),
            llm=LLMConfig(**config_dict.get('llm', {})),
            data=DataConfig(**config_dict.get('data', {}))
        )
