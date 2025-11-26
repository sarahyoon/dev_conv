"""
Main script for running cloud usage anomaly detection.

This script demonstrates how to use the anomaly detection framework
with both mock data and real data.
"""

import logging
import argparse
from pathlib import Path

from anomaly_detector.pipeline import AnomalyDetectionPipeline
from anomaly_detector.config import Config


def setup_logging(verbose: bool = False):
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('anomaly_detection.log')
        ]
    )


def main():
    """Main entry point for the anomaly detection application."""
    parser = argparse.ArgumentParser(
        description='Cloud Usage Anomaly Detection Framework'
    )
    parser.add_argument(
        '--customer-id',
        type=int,
        default=900,
        help='Customer ID to analyze (default: 900)'
    )
    parser.add_argument(
        '--use-mock-data',
        action='store_true',
        help='Use mock data instead of real data'
    )
    parser.add_argument(
        '--use-llm-api',
        action='store_true',
        help='Use LLM API for generating explanations'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--save-plot',
        type=str,
        help='Path to save the visualization plot'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        help='Path to CSV data file (if not using mock data)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Cloud Usage Anomaly Detection")
    
    # Create configuration
    config = Config()
    config.data.customer_id = args.customer_id
    
    # Initialize pipeline
    pipeline = AnomalyDetectionPipeline(config)
    
    # Load data
    df = None
    if not args.use_mock_data and args.data_file:
        import pandas as pd
        logger.info(f"Loading data from {args.data_file}")
        df = pd.read_csv(args.data_file, parse_dates=['Date'])
    
    # Run pipeline
    try:
        anomaly_df, explained_anomalies = pipeline.run(
            df=df,
            use_mock_data=args.use_mock_data or df is None,
            use_llm_api=args.use_llm_api,
            verbose=1 if args.verbose else 0,
            save_plot=args.save_plot
        )
        
        # Print results
        print("\n" + "=" * 60)
        print("ANOMALY DETECTION RESULTS")
        print("=" * 60)
        print(f"\nTotal anomalies detected: {len(anomaly_df[anomaly_df['anomaly_score'] > 0])}")
        print(f"Strong anomalies (score=1.0): {len(explained_anomalies)}")
        
        if not explained_anomalies.empty:
            print("\nStrong Anomalies with Explanations:")
            print("-" * 60)
            for date, row in explained_anomalies.iterrows():
                print(f"\nDate: {date.strftime('%Y-%m-%d')}")
                print(f"  Anomaly Score: {row['anomaly_score']:.1f}")
                print(f"  Cost: ${row['TotalCost']:.2f}")
                print(f"  Change: {row['pct_change']:.1f}%")
                print(f"  Explanation: {row.get('explanation', 'N/A')}")
        
        print("\n" + "=" * 60)
        
        # Save results to CSV
        output_file = f"anomaly_results_customer_{args.customer_id}.csv"
        anomaly_df.to_csv(output_file)
        logger.info(f"Results saved to {output_file}")
        
        if not explained_anomalies.empty:
            explained_file = f"strong_anomalies_customer_{args.customer_id}.csv"
            explained_anomalies.to_csv(explained_file)
            logger.info(f"Strong anomalies saved to {explained_file}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1
    
    logger.info("Anomaly detection completed successfully")
    return 0


if __name__ == '__main__':
    exit(main())
