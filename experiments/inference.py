import os
import sys
import json
import torch
import joblib
import pandas as pd
import logging
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from federated_learning import DataLoader
from gnn_models import TemporalGATDetector, ContentGATDetector, BehavioralGATDetector
from utils import plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load full dataset
    csv_path = 'datasets/fouad/NF-ToN-IoT-common-attacks.csv'
    logger.info(f"Loading full dataset from {csv_path}")
    df = pd.read_csv(csv_path)

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

    # We will use the temporal DataLoader purely as a utility to process the graph
    # Data loaders are stateful but we just want its processing pipeline
    data_dir = 'data/temporal_detector'
    if not os.path.exists(data_dir):
        logger.error(f"Data dir {data_dir} does not exist. Run experiment first.")
        return

    loader = DataLoader(data_dir, 'temporal', 'leiden')
    
    # Force load global label mapper
    loader._create_label_mapper(df)
    num_classes = len(loader.label_mapper)
    
    logger.info("Extracting features (this may take a few minutes for 1M rows)...")
    df = loader.feature_engineer.extract_features(df)
    df = loader.centrality_extractor.extract_centrality_features(df)
    
    logger.info("Extracting community features (Leiden)...")
    df = loader.community_processor.create_community_enhanced_features(df, {})

    logger.info("Converting to graph format...")
    graph_data = loader._process_to_graph(df)
    
    x = graph_data['features'].to(device)
    edge_index = graph_data['edge_index'].to(device)
    edge_labels = graph_data['edge_labels'].to(device)
    
    input_dim = x.shape[1]
    
    # Load Models
    detector_types = ['temporal', 'content', 'behavioral']
    models = {}
    
    for dtype in detector_types:
        model_path = f"results/{dtype}_model.pt"
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found. Run experiment first.")
            return
            
        if dtype == 'temporal':
            model = TemporalGATDetector(input_dim, 256, num_classes=num_classes)
        elif dtype == 'content':
            model = ContentGATDetector(input_dim, 256, num_classes=num_classes)
        elif dtype == 'behavioral':
            model = BehavioralGATDetector(input_dim, 256, num_classes=num_classes)
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models[dtype] = model
        logger.info(f"Loaded {dtype} model from {model_path}")

    # Process through GNNs
    all_probs = []
    logger.info("Running full graph through GNN detectors...")
    with torch.no_grad():
        for dtype in detector_types:
            logger.info(f"Running {dtype} detector...")
            _, edge_predictions = models[dtype](x, edge_index)
            probs = torch.softmax(edge_predictions, dim=1)
            all_probs.append(probs)

    ensemble_features = torch.cat(all_probs, dim=1).cpu().numpy()
    y_true = edge_labels.cpu().numpy()

    # Load Random Forest
    rf_path = "results/rf_model.joblib"
    if not os.path.exists(rf_path):
        logger.error(f"RF model {rf_path} not found. Run experiment first.")
        return
        
    rf_model = joblib.load(rf_path)
    logger.info(f"Loaded Random Forest model from {rf_path}")

    # Predict
    logger.info("Predicting final labels using Random Forest...")
    y_pred = rf_model.predict(ensemble_features)

    # Metrics
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    logger.info("=========================================")
    logger.info(f"FULL DATASET INFERENCE COMPLETE")
    logger.info(f"Total Flows Evaluated: {len(y_true)}")
    logger.info(f"Balanced Accuracy: {bal_acc:.4f}")
    logger.info(f"Standard Accuracy: {acc:.4f}")
    logger.info(f"Macro F1 Score:    {macro_f1:.4f}")
    logger.info("=========================================")

    # Plot confusion matrix
    class_names = [k for k, v in sorted(loader.label_mapper.items(), key=lambda item: item[1])]
    cm_path = "results/full_dataset_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    logger.info(f"Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    main()
