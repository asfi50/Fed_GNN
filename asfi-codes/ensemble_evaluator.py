"""
Random Forest Ensemble Evaluator for FedGATSage
================================================

Overview of what this file does, top to bottom:

STEP 1 — Feed validation CSV through all 3 GATs.
          Each GAT outputs softmax probabilities for each network flow.

STEP 2 — Build a new "meta-dataset" from those GAT outputs only (not the raw CSV).
          For each flow, the feature vector contains:
            - Temporal GAT   : [class probs] + [predicted class] + [confidence score]
            - Content GAT    : [class probs] + [predicted class] + [confidence score]
            - Behavioral GAT : [class probs] + [predicted class] + [confidence score]
            - Ensemble avg   : [avg probs]   + [avg class]       + [avg confidence]

STEP 3 — Train the Random Forest on this meta-dataset (val set only).

STEP 4 — Feed the TEST CSV through the same 3 GATs to build a separate meta-dataset.

STEP 5 — Evaluate the Random Forest on the test meta-dataset. Log all metrics to CometML.
"""

import os
import sys
import comet_ml
import torch
import numpy as np
import pandas as pd
import logging
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)

# Add src/ to path so we can import plot_confusion_matrix
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)
from utils import plot_confusion_matrix


class RandomForestEnsembleEvaluator:
    """
    Trains and evaluates a Random Forest that fuses the outputs of the
    Temporal, Content, and Behavioral GAT detectors, as described in the paper.
    """
    
    def __init__(self, fed_system, args):
        self.fed_system = fed_system
        self.device = fed_system.device
        self.args = args
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # -------------------------------------------------------------------------
    # PRIVATE HELPERS
    # -------------------------------------------------------------------------

    def _load_and_process_csv(self, csv_path: str, test_loader) -> dict:
        """Load a CSV file and run the same feature engineering pipeline used during training."""
        df = pd.read_csv(csv_path)
        if self.args.demo_mode:
            df = df.head(1000)
        df = test_loader.feature_engineer.extract_features(df)
        df = test_loader.centrality_extractor.extract_centrality_features(df)
        df = test_loader.community_processor.create_community_enhanced_features(df, {})
        return test_loader._process_to_graph(df)

    def _run_gats_on_graph(self, graph_data: dict) -> tuple:
        """
        Feed a processed graph through all 3 GATs.
        
        Returns:
            X           : numpy array of shape (n_flows, n_meta_features)
                          Each row is the ensemble feature vector for one network flow.
            y           : numpy array of ground-truth class labels (int)
            gat_probs   : dict mapping detector_type -> raw softmax probs (numpy array)
        """
        x = graph_data['features'].to(self.device)
        edge_index = graph_data['edge_index'].to(self.device)
        y = graph_data['edge_labels'].cpu().numpy()

        detector_order = ['temporal', 'content', 'behavioral']
        all_probs = []   # one entry per detector (torch tensor on device)
        gat_probs = {}   # detector_type -> numpy array, for individual evaluation

        for detector_type in detector_order:
            if detector_type not in self.fed_system.client_models:
                logger.warning(f"Detector '{detector_type}' not found. Filling with zeros.")
                num_classes = list(self.fed_system.client_models.values())[0][0].classifiers[0].out_features
                zeros = torch.zeros((len(y), num_classes)).to(self.device)
                all_probs.append(zeros)
                gat_probs[detector_type] = zeros.cpu().numpy()
                continue

            # Use client 0's model for inference (it holds the federated-averaged weights)
            model = self.fed_system.client_models[detector_type][0]
            model.eval()
            with torch.no_grad():
                _, logits = model(x, edge_index)
                probs = torch.softmax(logits, dim=1)
            all_probs.append(probs)
            gat_probs[detector_type] = probs.cpu().numpy()

        # Build the meta-feature matrix as described in the paper:
        # [per-GAT probs | predicted class | confidence] + [ensemble avg probs | avg class | avg confidence]
        feature_parts = []
        for probs in all_probs:
            probs_np = probs.cpu().numpy()
            pred_class = np.argmax(probs_np, axis=1, keepdims=True)   # shape (n, 1)
            confidence = np.max(probs_np, axis=1, keepdims=True)      # shape (n, 1)
            feature_parts.extend([probs_np, pred_class, confidence])

        avg_probs = torch.stack(all_probs).mean(dim=0).cpu().numpy()
        avg_pred_class = np.argmax(avg_probs, axis=1, keepdims=True)
        avg_confidence = np.max(avg_probs, axis=1, keepdims=True)
        feature_parts.extend([avg_probs, avg_pred_class, avg_confidence])

        X = np.hstack(feature_parts)
        # Sanitize for sklearn (no NaN / inf / float overflow)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, -np.finfo(np.float32).max, np.finfo(np.float32).max)

        return X, y, gat_probs

    def _evaluate_individual_gats(self, gat_probs: dict, y_true: np.ndarray, class_names: list):
        """Log per-GAT metrics and save their confusion matrices."""
        order_map = {'temporal': '2', 'content': '4', 'behavioral': '3'}
        
        for detector_type, probs in gat_probs.items():
            y_pred = np.argmax(probs, axis=1)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')

            logger.info(
                f"{detector_type.capitalize()} GAT — "
                f"Balanced Acc: {bal_acc:.4f} | Acc: {acc:.4f} | Macro F1: {f1:.4f}"
            )

            # Log to CometML
            try:
                exp = comet_ml.get_global_experiment()
                if exp:
                    order = order_map.get(detector_type, '5')
                    exp.log_metrics({
                        f"Test/{order}_{detector_type.capitalize()}_Balanced_Accuracy": bal_acc,
                        f"Test/{order}_{detector_type.capitalize()}_Standard_Accuracy": acc,
                        f"Test/{order}_{detector_type.capitalize()}_Macro_F1": f1,
                    })
            except Exception as e:
                logger.warning(f"CometML log failed for {detector_type}: {e}")

            # Save confusion matrix PNG + CSV
            if class_names:
                try:
                    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
                    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
                    
                    png_path = os.path.join(self.args.output_dir, f'{detector_type}_confusion_matrix.png')
                    csv_path = os.path.join(self.args.output_dir, f'{detector_type}_confusion_matrix.csv')
                    
                    plot_confusion_matrix(y_true, y_pred, class_names, png_path)
                    cm_df.to_csv(csv_path)
                    logger.info(f"{detector_type.capitalize()} confusion matrix saved: PNG + CSV")
                except Exception as e:
                    logger.warning(f"Could not save confusion matrix for {detector_type}: {e}")

    # -------------------------------------------------------------------------
    # PUBLIC METHOD — MAIN ENTRY POINT
    # -------------------------------------------------------------------------

    def evaluate(self, test_data_path: str, test_loader) -> dict:
        """
        Full evaluation pipeline:
          1. Run val.csv through all 3 GATs  -> train the Random Forest
          2. Run test.csv through all 3 GATs -> evaluate the Random Forest
          3. Log all metrics and save confusion matrices
        """
        logger.info("=" * 60)
        logger.info("Starting Random Forest Ensemble Evaluation")
        logger.info("=" * 60)

        class_names = None
        if test_loader.label_mapper:
            class_names = [k for k, v in sorted(test_loader.label_mapper.items(), key=lambda x: x[1])]

        # ------------------------------------------------------------------ #
        # STEP 1 & 2: Build the RF training set from val.csv                 #
        # ------------------------------------------------------------------ #
        val_data_path = test_data_path.replace('test.csv', 'val.csv')
        
        if not os.path.exists(val_data_path):
            logger.warning(
                f"val.csv not found at '{val_data_path}'. "
                "Re-run preprocess_data.py to generate it. "
                "Falling back to 50/50 split of test.csv (less rigorous)."
            )
            # Graceful fallback: split the test set itself
            graph_test = self._load_and_process_csv(test_data_path, test_loader)
            if graph_test is None or len(graph_test['edge_labels']) == 0:
                logger.error("Test graph is empty. Aborting evaluation.")
                return {}
            X_all, y_all, gat_probs = self._run_gats_on_graph(graph_test)
            mid = len(y_all) // 2
            X_val_meta, y_val_meta = X_all[:mid], y_all[:mid]
            X_test_meta, y_test_meta = X_all[mid:], y_all[mid:]
            gat_probs_test = gat_probs
        else:
            logger.info(f"[STEP 1/5] Loading and processing val.csv: {val_data_path}")
            graph_val = self._load_and_process_csv(val_data_path, test_loader)
            if graph_val is None or len(graph_val['edge_labels']) == 0:
                logger.error("Validation graph is empty. Aborting evaluation.")
                return {}

            logger.info("[STEP 2/5] Running all 3 GATs on validation data to build RF training set")
            X_val_meta, y_val_meta, _ = self._run_gats_on_graph(graph_val)

            logger.info(f"[STEP 3/5] Loading and processing test.csv: {test_data_path}")
            graph_test = self._load_and_process_csv(test_data_path, test_loader)
            if graph_test is None or len(graph_test['edge_labels']) == 0:
                logger.error("Test graph is empty. Aborting evaluation.")
                return {}

            logger.info("[STEP 4/5] Running all 3 GATs on test data to build RF evaluation set")
            X_test_meta, y_test_meta, gat_probs_test = self._run_gats_on_graph(graph_test)

        # ------------------------------------------------------------------ #
        # STEP 3: Evaluate individual GATs on the TEST set                   #
        # ------------------------------------------------------------------ #
        logger.info("[STEP 5/5] Evaluating individual GATs and training + evaluating the Random Forest")
        logger.info("--- Individual GAT Performance (on test set) ---")
        self._evaluate_individual_gats(gat_probs_test, y_test_meta, class_names)

        # ------------------------------------------------------------------ #
        # STEP 4: Train RF on val meta-dataset, evaluate on test meta-dataset#
        # ------------------------------------------------------------------ #
        logger.info(f"--- Training Random Forest on {len(y_val_meta)} validation meta-features ---")
        self.rf_model.fit(X_val_meta, y_val_meta)

        logger.info(f"--- Evaluating Random Forest on {len(y_test_meta)} test meta-features ---")
        y_pred = self.rf_model.predict(X_test_meta)

        bal_acc = balanced_accuracy_score(y_test_meta, y_pred)
        acc = accuracy_score(y_test_meta, y_pred)
        f1 = f1_score(y_test_meta, y_pred, average='macro')

        logger.info(
            f"Ensemble (RF) Result — "
            f"Balanced Acc: {bal_acc:.4f} | Acc: {acc:.4f} | Macro F1: {f1:.4f}"
        )

        # ------------------------------------------------------------------ #
        # STEP 5: Save confusion matrices and log to CometML                  #
        # ------------------------------------------------------------------ #
        if class_names:
            try:
                cm = confusion_matrix(y_test_meta, y_pred, labels=range(len(class_names)))
                cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
                png_path = os.path.join(self.args.output_dir, 'confusion_matrix.png')
                csv_path = os.path.join(self.args.output_dir, 'confusion_matrix.csv')
                plot_confusion_matrix(y_test_meta, y_pred, class_names, png_path)
                cm_df.to_csv(csv_path)
                logger.info(f"Ensemble confusion matrix saved: PNG + CSV")
            except Exception as e:
                logger.warning(f"Could not save ensemble confusion matrix: {e}")

        try:
            rf_path = os.path.join(self.args.output_dir, 'rf_model.joblib')
            joblib.dump(self.rf_model, rf_path)
            logger.info(f"Random Forest model saved to {rf_path}")
        except Exception as e:
            logger.warning(f"Could not save RF model: {e}")

        try:
            exp = comet_ml.get_global_experiment()
            if exp:
                exp.log_metrics({
                    "Test/1_Balanced_Accuracy": bal_acc,
                    "Test/1_Standard_Accuracy": acc,
                    "Test/1_Macro_F1": f1,
                })
                if class_names:
                    exp.log_confusion_matrix(
                        y_true=y_test_meta.tolist(),
                        y_predicted=y_pred.tolist(),
                        labels=class_names
                    )
        except Exception as e:
            logger.warning(f"CometML log failed for ensemble: {e}")

        return {'accuracy': acc, 'balanced_accuracy': bal_acc, 'macro_f1': f1}
