"""
Visualization module for anomaly detection results.

This module provides functions for visualizing time series data and anomaly detection results.
"""

from typing import Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging


logger = logging.getLogger(__name__)


class AnomalyVisualizer:
    """Visualizes anomaly detection results."""
    
    @staticmethod
    def plot_anomaly_results(
        anomaly_df: pd.DataFrame,
        strong_anomalies: pd.DataFrame,
        customer_id: int,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a dual-axis plot showing total cost and anomaly scores.
        
        Args:
            anomaly_df: DataFrame with anomaly detection results.
            strong_anomalies: DataFrame with strong anomalies only.
            customer_id: Customer ID for the plot title.
            figsize: Figure size (width, height).
            save_path: Optional path to save the figure.
            
        Returns:
            plt.Figure: The created figure.
        """
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Plot Total Cost on primary axis
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Total Cost', color=color)
        ax1.plot(
            anomaly_df.index,
            anomaly_df['TotalCost'],
            color=color,
            label='Total Cost',
            linewidth=2
        )
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # Highlight strong anomalies on the cost line
        if not strong_anomalies.empty:
            ax1.scatter(
                strong_anomalies.index,
                strong_anomalies['TotalCost'],
                color='red',
                s=100,
                zorder=5,
                label='Strong Anomaly (Score 1.0)',
                marker='X'
            )
        
        # Plot Anomaly Score on secondary axis
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Anomaly Score (0.0, 0.5, 1.0)', color=color)
        ax2.plot(
            anomaly_df.index,
            anomaly_df['anomaly_score'],
            color=color,
            linestyle='--',
            marker='o',
            label='Anomaly Score',
            alpha=0.7
        )
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_yticks([0.0, 0.5, 1.0])
        ax2.set_ylim(-0.1, 1.1)
        
        # Title and legend
        plt.title(
            f'고객 {customer_id} 하이브리드 이상 탐지 결과 (Anomaly Score)',
            fontsize=14,
            pad=20
        )
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        logger.info("Generated anomaly results plot")
        
        return fig
    
    @staticmethod
    def plot_training_history(
        history,
        figsize: Tuple[int, int] = (10, 5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training and validation loss over epochs.
        
        Args:
            history: Keras History object from model training.
            figsize: Figure size (width, height).
            save_path: Optional path to save the figure.
            
        Returns:
            plt.Figure: The created figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('LSTM Autoencoder Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        logger.info("Generated training history plot")
        
        return fig
    
    @staticmethod
    def plot_reconstruction_error_distribution(
        train_errors: np.ndarray,
        test_errors: np.ndarray,
        threshold: float,
        figsize: Tuple[int, int] = (10, 5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution of reconstruction errors for train and test sets.
        
        Args:
            train_errors: Reconstruction errors for training data.
            test_errors: Reconstruction errors for test data.
            threshold: Anomaly detection threshold.
            figsize: Figure size (width, height).
            save_path: Optional path to save the figure.
            
        Returns:
            plt.Figure: The created figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histograms
        ax.hist(
            train_errors,
            bins=50,
            alpha=0.5,
            label='Training Errors',
            color='blue',
            density=True
        )
        ax.hist(
            test_errors,
            bins=50,
            alpha=0.5,
            label='Test Errors',
            color='green',
            density=True
        )
        
        # Plot threshold line
        ax.axvline(
            threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Threshold ({threshold:.4f})'
        )
        
        ax.set_xlabel('Reconstruction Error (MSE)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Reconstruction Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Error distribution plot saved to {save_path}")
        
        logger.info("Generated reconstruction error distribution plot")
        
        return fig
    
    @staticmethod
    def plot_time_series_with_anomalies(
        time_series: pd.Series,
        anomaly_dates: pd.DatetimeIndex,
        title: str = 'Time Series with Detected Anomalies',
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot time series with anomaly points highlighted.
        
        Args:
            time_series: Time series data to plot.
            anomaly_dates: Dates where anomalies were detected.
            title: Plot title.
            figsize: Figure size (width, height).
            save_path: Optional path to save the figure.
            
        Returns:
            plt.Figure: The created figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot time series
        ax.plot(
            time_series.index,
            time_series.values,
            label='Total Cost',
            linewidth=2,
            color='blue'
        )
        
        # Highlight anomalies
        if len(anomaly_dates) > 0:
            anomaly_values = time_series.loc[anomaly_dates]
            ax.scatter(
                anomaly_dates,
                anomaly_values,
                color='red',
                s=100,
                zorder=5,
                label='Detected Anomalies',
                marker='X'
            )
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Total Cost')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Time series plot saved to {save_path}")
        
        logger.info("Generated time series plot with anomalies")
        
        return fig
