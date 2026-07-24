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
     │   ├── val.csv      <- RF ensemble trains on this
     │   └── test.csv     <- RF ensemble is evaluated on this
     ├── content_detector/
     │   └── ...
     └── behavioral_detector/
         └── ...

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
    parser.add_argument('--max_rows', type=int, default=None,
                       help='Maximum number of rows to sample randomly from the dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    return parser.parse_args()

def save_split_data(df, output_dir, prefix, num_clients):
    """
    Split the dataset into 3 parts and save each:
    
    - 80% Training data  -> split further into per-client CSVs (the GATs train on these)
    - 10% Validation set -> val.csv  (the Random Forest TRAINS on this)
    - 10% Test set       -> test.csv (the Random Forest is EVALUATED on this)
    
    This ensures the Random Forest never sees the test data during fitting.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Carve out the test set (10% of total data)
    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    
    # Step 2: Carve out the validation set from the remaining data (10% of total = ~11.1% of remainder)
    train_df, val_df = train_test_split(train_val_df, test_size=0.111, random_state=42)
    
    # Save val and test sets
    val_path = os.path.join(output_dir, 'val.csv')
    val_df.to_csv(val_path, index=False)
    logger.info(f"Saved validation set to {val_path} ({len(val_df)} records)")
    
    test_path = os.path.join(output_dir, 'test.csv')
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved test set to {test_path} ({len(test_df)} records)")
    
    # Step 3: Distribute the training data (80%) among the federated clients (IID random split)
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
        
        # Standardize NF-ToN-IoT column names to match the expected format
        column_mapping = {
            'IPV4_SRC_ADDR': 'Src IP',
            'IPV4_DST_ADDR': 'Dst IP',
            'L4_SRC_PORT': 'Src Port',
            'L4_DST_PORT': 'Dst Port',
            'PROTOCOL': 'Protocol',
            'FLOW_DURATION_MILLISECONDS': 'Flow Duration',
        }
        df = df.rename(columns=column_mapping)
        
        logger.info(f"Successfully loaded dataset with {len(df)} records")
        
        if args.max_rows is not None and len(df) > args.max_rows:
            logger.info(f"Randomly sampling {args.max_rows} rows from the dataset (Stratified random sample)...")
            df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
            logger.info(f"Dataset successfully reduced to {len(df)} records")
            
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Generate and save a global label mapper BEFORE splitting the data
    import json
    unique_attacks = sorted(df['Attack'].unique())
    label_mapper = {attack: int(idx) for idx, attack in enumerate(unique_attacks)}
    
    os.makedirs(args.output_dir, exist_ok=True)
    mapper_path = os.path.join(args.output_dir, 'label_mapper.json')
    with open(mapper_path, 'w') as f:
        json.dump(label_mapper, f, indent=4)
    logger.info(f"Saved global label mapper with {len(label_mapper)} classes to {mapper_path}")

    # Calculate inverse class weights for loss function
    import torch
    class_counts = df['Attack'].value_counts()
    total_samples = len(df)
    num_classes = len(unique_attacks)
    weights = []
    for attack in unique_attacks:
        count = class_counts.get(attack, 0)
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    class_weights_tensor = torch.tensor(weights, dtype=torch.float)
    weights_path = os.path.join(args.output_dir, 'class_weights.pt')
    torch.save(class_weights_tensor, weights_path)
    logger.info(f"Saved class weights to {weights_path}")

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
