"""
Data Preprocessing Script for FedGATSage
========================================

This script prepares raw network traffic data (CSV format) for the FedGATSage experiment.
It performs the following steps:
1. Loads the raw dataset (e.g., CIC-ToN-IoT).
2. Splits the data into training and testing sets.
3. Partitions the training data among federated clients.
4. Organizes the data into the directory structure expected by the experiment script:
   data/
     ├── temporal_detector/
     │   ├── client_1.csv
     │   ├── ...
     │   └── test.csv
     ├── content_detector/
     │   ├── client_1.csv
     │   ├── ...
     │   └── test.csv
     └── behavioral_detector/
         ├── client_1.csv
         ├── ...
         └── test.csv

Usage:
    python preprocess_data.py --input_file path/to/dataset.csv --output_dir data --num_clients 5
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='FedGATSage Data Preprocessing')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to the raw CSV dataset')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Directory to save processed data')
    parser.add_argument('--num_clients', type=int, default=5,
                       help='Number of federated clients')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Ratio of data to use for testing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    return parser.parse_args()

def save_split_data(df, output_dir, prefix, num_clients):
    """
    Split the dataset and save it into client-specific files.
    
    We create:
    1. A test set (20% of data) for evaluation.
    2. Individual client files for the remaining 80% (training data).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # First, let's set aside some data for testing
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save the test set
    test_path = os.path.join(output_dir, 'test.csv')
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved test set to {test_path} ({len(test_df)} records)")
    
    # Now, distribute the training data among the clients
    # In this reference implementation, we use an IID split (random shuffle).
    # For more advanced scenarios, you might want to implement non-IID splitting 
    # (e.g., by attack type or IP range) to simulate real-world heterogeneity.
    client_dfs = np.array_split(train_df, num_clients)
    
    for i, client_df in enumerate(client_dfs):
        client_id = i + 1
        client_path = os.path.join(output_dir, f'client_{client_id}.csv')
        client_df.to_csv(client_path, index=False)
        logger.info(f"Saved client {client_id} data to {client_path} ({len(client_df)} records)")

def main():
    args = parse_args()
    
    logger.info(f"Starting preprocessing with input: {args.input_file}")
    
    if not os.path.exists(args.input_file):
        logger.error(f"Could not find the input file: {args.input_file}")
        raise FileNotFoundError(f"Input file not found: {args.input_file}. Please check the path and try again.")
    
    # Load the dataset
    try:
        # We load the full dataset here. If your dataset is massive, you might want to 
        # implement chunked reading or use a library like Dask.
        df = pd.read_csv(args.input_file)
        logger.info(f"Successfully loaded dataset with {len(df)} records")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # We process data for each detector type.
    # In this reference implementation, we distribute the full dataset to all detectors.
    # The specialized logic for each detector (Temporal vs Content vs Behavioral) 
    # happens during feature extraction in the experiment phase.
    
    detector_types = ['temporal', 'content', 'behavioral']
    
    for detector in detector_types:
        logger.info(f"Preparing data for the {detector} detector...")
        detector_dir = os.path.join(args.output_dir, f'{detector}_detector')
        
        save_split_data(df, detector_dir, detector, args.num_clients)
        
    logger.info("All done! Data preprocessing is complete.")

if __name__ == "__main__":
    main()
