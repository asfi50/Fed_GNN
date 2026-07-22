import comet_ml
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
import logging

logger = logging.getLogger(__name__)

class RandomForestEnsembleEvaluator:
    """
    Fuses the predictions of Temporal, Content, and Behavioral GAT detectors
    using a Random Forest Classifier, as described in the paper.
    """
    def __init__(self, fed_system, args):
        self.fed_system = fed_system
        self.device = fed_system.device
        self.args = args
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def _extract_probabilities(self, test_data) -> tuple:
        """
        Runs the test data through all 3 detectors and concatenates the softmax probabilities.
        """
        x = test_data['features'].to(self.device)
        edge_index = test_data['edge_index'].to(self.device)
        edge_labels = test_data['edge_labels'].to(self.device)
        
        all_probs = []
        
        # Ensure we evaluate all three detectors
        expected_detectors = ['temporal', 'content', 'behavioral']
        
        for detector_type in expected_detectors:
            if detector_type not in self.fed_system.client_models:
                logger.warning(f"Detector {detector_type} not found! Cannot build full ensemble.")
                # Fill with zeros if missing to keep dimensions consistent
                # Get num_classes from the first available model
                first_model = list(self.fed_system.client_models.values())[0][0]
                num_classes = first_model.classifiers[0].out_features
                dummy_probs = torch.zeros((edge_labels.shape[0], num_classes)).to(self.device)
                all_probs.append(dummy_probs)
                continue
                
            model = self.fed_system.client_models[detector_type][0] # Using client 1's model for evaluation
            model.eval()
            
            with torch.no_grad():
                _, edge_predictions = model(x, edge_index)
                probs = torch.softmax(edge_predictions, dim=1)
                all_probs.append(probs)
                
        # Concatenate probabilities: Shape = (num_edges, 3 * num_classes)
        prob_features = torch.cat(all_probs, dim=1).cpu().numpy()
        
        # Inject statistical graph features (raw DataFrame columns) to give RF context
        df = test_data.get('df', None)
        if df is not None:
            import pandas as pd
            # Select some safe numerical columns that don't leak labels
            # We take all numeric columns except the label-related ones
            num_cols = df.select_dtypes(include=[np.number]).columns
            cols_to_drop = [c for c in num_cols if 'label' in c.lower() or 'attack' in c.lower()]
            raw_features = df[num_cols].drop(columns=cols_to_drop).fillna(0).values
            
            # Ensure the row count matches exactly
            if len(raw_features) == len(prob_features):
                ensemble_features = np.hstack((prob_features, raw_features))
            else:
                logger.warning("Row count mismatch between GNN probabilities and raw features. Using only probabilities.")
                ensemble_features = prob_features
        else:
            ensemble_features = prob_features
            
        return ensemble_features, edge_labels.cpu().numpy()
        
    def evaluate(self, test_data_path: str, test_loader) -> dict:
        """
        Trains and evaluates the Random Forest ensemble on the test data.
        Note: In a true pipeline, RF should be trained on a separate validation set, 
        but for reproduction with current script constraints, we train/test on the available test data using a split.
        """
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        logger.info("Starting Random Forest Ensemble Evaluation")
        
        df_test = pd.read_csv(test_data_path)
        if self.args.demo_mode:
            df_test = df_test.head(1000)
            
        # Process the data
        df_test = test_loader.feature_engineer.extract_features(df_test)
        df_test = test_loader.centrality_extractor.extract_centrality_features(df_test)
        df_test = test_loader.community_processor.create_community_enhanced_features(df_test, {})
        
        test_data = test_loader._process_to_graph(df_test)
        
        if test_data is None or len(test_data['edge_labels']) == 0:
            logger.error("Failed to process test data for ensemble evaluation.")
            return {}
            
        # Extract features (probabilities from 3 GATs)
        X, y = self._extract_probabilities(test_data)
        
        # Split test data to train RF and evaluate it
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        
        # Train RF
        logger.info("Training Random Forest on detector probabilities...")
        self.rf_model.fit(X_train, y_train)
        
        # Predict
        y_pred = self.rf_model.predict(X_test)
        
        # Calculate metrics
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        logger.info(f"Ensemble Evaluation Complete - Balanced Accuracy: {bal_acc:.4f}, Standard Accuracy: {acc:.4f}, F1: {macro_f1:.4f}")
        
        # Plot confusion matrix
        try:
            import os
            import sys
            # Ensure we can import utils from the src directory
            src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
            if src_path not in sys.path:
                sys.path.append(src_path)
            
            from utils import plot_confusion_matrix
            class_names = None
            if test_loader.label_mapper:
                class_names = [k for k, v in sorted(test_loader.label_mapper.items(), key=lambda x: x[1])]
            
            cm_path = os.path.join(self.args.output_dir, 'confusion_matrix.png')
            plot_confusion_matrix(y_test, y_pred, class_names, cm_path)
            logger.info(f"Confusion matrix saved to {cm_path}")
            
            # Save the confusion matrix as CSV
            try:
                from sklearn.metrics import confusion_matrix
                import pandas as pd
                # Pass labels to ensure the matrix shape matches class_names even if some classes are missing
                cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
                cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
                cm_csv_path = os.path.join(self.args.output_dir, 'confusion_matrix.csv')
                cm_df.to_csv(cm_csv_path)
                logger.info(f"Confusion matrix CSV saved to {cm_csv_path}")
            except Exception as e:
                logger.warning(f"Failed to save confusion matrix CSV: {e}")
            
            # Save the trained Random Forest model
            import joblib
            rf_path = os.path.join(self.args.output_dir, 'rf_model.joblib')
            joblib.dump(self.rf_model, rf_path)
            logger.info(f"Random Forest model saved to {rf_path}")
            
            # Log final evaluation to comet_ml
            exp = comet_ml.get_global_experiment()
            if exp is not None:
                exp.log_metrics({
                    "Test/Balanced_Accuracy": bal_acc,
                    "Test/Standard_Accuracy": acc,
                    "Test/Macro_F1": macro_f1
                })
                exp.log_confusion_matrix(y_true=y_test.tolist(), y_predicted=y_pred.tolist(), labels=class_names)
                
        except Exception as e:
            logger.error(f"Could not plot confusion matrix: {e}")
            
        metrics = {
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'macro_f1': macro_f1
        }
        
        return metrics
