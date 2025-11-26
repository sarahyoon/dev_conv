# Cloud Usage Anomaly Detection Framework

A robust, production-ready framework for detecting cost/usage anomalies in cloud usage data using LSTM Autoencoder and hybrid detection methods.

## Overview

This framework implements an advanced anomaly detection system that combines:
1. **LSTM Autoencoder** - Detects pattern deviations in time series data
2. **Decrease-Rate Detection** - Identifies significant cost decreases
3. **Hybrid Scoring** - Combines both methods for accurate anomaly detection
4. **LLM Explanations** - Generates natural language explanations for detected anomalies

## Features

- ✅ Modular, clean code architecture following SOLID principles
- ✅ Comprehensive error handling and input validation
- ✅ Type hints for better code clarity
- ✅ Detailed logging throughout the pipeline
- ✅ Configuration management with dataclasses
- ✅ Professional visualization tools
- ✅ Support for both mock and real data
- ✅ Optional LLM integration for anomaly explanations
- ✅ Command-line interface for easy usage

## Project Structure

```
dev_conv/
├── src/
│   ├── anomaly_detector/
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration management
│   │   ├── data_preprocessing.py  # Data loading and preprocessing
│   │   ├── model.py                # LSTM Autoencoder model
│   │   ├── detector.py             # Anomaly detection logic
│   │   ├── explainer.py            # LLM-based explanations
│   │   ├── visualization.py        # Visualization tools
│   │   └── pipeline.py             # Main pipeline orchestration
│   └── main.py                     # Command-line interface
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── notebooks/
    └── example_usage.ipynb         # Example Jupyter notebook
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

#### Using Mock Data (for testing)

```bash
python src/main.py --use-mock-data --customer-id 900 --save-plot results.png
```

#### Using Real Data

```bash
python src/main.py --data-file total_clean.csv --customer-id 500 --save-plot results.png
```

#### With LLM Explanations

```bash
export GEMINI_API_KEY="your-api-key-here"
python src/main.py --use-mock-data --use-llm-api
```

#### Verbose Mode

```bash
python src/main.py --use-mock-data --verbose
```

### Python API

```python
from anomaly_detector.pipeline import AnomalyDetectionPipeline
from anomaly_detector.config import Config
import pandas as pd

# Create configuration
config = Config()
config.data.customer_id = 900

# Initialize pipeline
pipeline = AnomalyDetectionPipeline(config)

# Option 1: Use mock data
anomaly_df, explained_anomalies = pipeline.run(use_mock_data=True)

# Option 2: Use real data
df = pd.read_csv('total_clean.csv', parse_dates=['Date'])
anomaly_df, explained_anomalies = pipeline.run(df=df, use_mock_data=False)

# Display results
print(explained_anomalies)
```

## Configuration

The framework uses a hierarchical configuration system with the following components:

### ModelConfig
- `window_size`: Number of days in input sequence (default: 7)
- `lstm_units`: Number of LSTM units (default: 32)
- `epochs`: Maximum training epochs (default: 50)
- `batch_size`: Training batch size (default: 16)

### DetectionConfig
- `split_ratio`: Train/test split ratio (default: 0.8)
- `decrease_threshold`: Threshold for cost decrease (default: -0.3)
- `lstm_sigma_multiplier`: Threshold multiplier (default: 1.5)

### LLMConfig
- `api_key`: Gemini API key (from environment or config)
- `max_explanation_length`: Max characters in explanation (default: 70)

### DataConfig
- `customer_id`: Target customer ID (default: 900)
- `date_column`: Date column name (default: 'Date')
- `cost_column`: Cost column name (default: 'TotalCost')

## Architecture

### Data Flow

1. **Data Loading** → Load and validate time series data
2. **Preprocessing** → Normalize and create sequences
3. **Model Training** → Train LSTM Autoencoder
4. **LSTM Detection** → Calculate reconstruction errors
5. **Decrease Detection** → Identify significant decreases
6. **Hybrid Scoring** → Combine detection methods
7. **Explanation** → Generate natural language explanations
8. **Visualization** → Create plots and reports

### Key Classes

- `AnomalyDetectionPipeline`: Main orchestration class
- `LSTMAutoencoder`: LSTM Autoencoder model
- `AnomalyDetector`: Hybrid detection logic
- `ExplanationGenerator`: LLM-based explanations
- `DataPreprocessor`: Data handling and preprocessing
- `AnomalyVisualizer`: Visualization utilities

## Clean Code Principles Applied

1. **Single Responsibility Principle**: Each module has one clear purpose
2. **Separation of Concerns**: Data, model, detection, and visualization are separate
3. **DRY (Don't Repeat Yourself)**: Common functionality is abstracted
4. **Type Hints**: All functions include type annotations
5. **Docstrings**: Comprehensive documentation for all classes and functions
6. **Error Handling**: Proper exception handling with informative messages
7. **Logging**: Structured logging instead of print statements
8. **Configuration Management**: Centralized, type-safe configuration
9. **Testability**: Modular design enables easy unit testing

## Security Considerations

- ✅ API keys loaded from environment variables
- ✅ Input validation on all user-provided data
- ✅ No hardcoded sensitive information
- ✅ Proper error messages without exposing internals

## Performance Optimization

- Vectorized operations with NumPy
- Efficient data structures (pandas DataFrame)
- Early stopping in model training
- Batch processing for predictions

## Extensibility

The framework is designed to be easily extended:

- Add new detection methods in `detector.py`
- Implement different model architectures in `model.py`
- Add custom visualization in `visualization.py`
- Integrate different LLM providers in `explainer.py`

## Testing

Run tests with pytest (when tests are implemented):

```bash
pytest src/tests/ -v --cov=anomaly_detector
```

## Original Notebook

The original implementation can be found in `proj.ipynb`. The refactored code improves upon it by:

- Extracting monolithic notebook into modular Python files
- Adding proper error handling and validation
- Implementing clean code principles
- Adding type hints and comprehensive documentation
- Creating reusable components
- Improving maintainability and testability

## Contributing

When contributing, please:
1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Write comprehensive docstrings
4. Include unit tests for new functionality
5. Update documentation as needed

## License

This project is provided as-is for educational and commercial use.

## Contact

For questions or issues, please open a GitHub issue in the repository.
