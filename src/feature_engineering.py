"""
Feature engineering for FedGATSage specialized detectors.
Extracts community-aware features for temporal, content, and behavioral attack detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Handles the extraction of specialized features for each GAT detector.
    This ensures that the Temporal, Content, and Behavioral models each get the 
    data they need to excel at their specific tasks.
    """
    
    def __init__(self, detector_type='temporal'):
        self.detector_type = detector_type
        self.created_features = []
        
        # We group attacks to ensure the right features are generated for the right problem
        self.temporal_attacks = ['ddos', 'dos', 'scanning']
        self.content_attacks = ['injection', 'xss'] 
        self.behavioral_attacks = ['password', 'backdoor', 'ransomware']
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point: takes raw data and adds the specialized columns 
        needed for the current detector type.
        """
        result_df = df.copy()
        
        # First, everyone gets the basics (flow rates, payload sizes)
        result_df = self._add_base_features(result_df)
        
        # Then we add the specialized features
        if self.detector_type == 'temporal':
            result_df = self._add_temporal_features(result_df)
        elif self.detector_type == 'content':
            result_df = self._add_content_features(result_df)
        elif self.detector_type == 'behavioral':
            result_df = self._add_behavioral_features(result_df)
            
        logger.info(f"Engineered {len(self.created_features)} new features for {self.detector_type} detection")
        return result_df
    
    def _add_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add base traffic features for all detector types"""
        
        # Flow rate features
        if 'Flow Duration' in df.columns and 'Tot Fwd Pkts' in df.columns:
            df['flow_rate'] = (df['Tot Fwd Pkts'] + df['Tot Bwd Pkts']) / (df['Flow Duration'] / 1000000 + 1e-6)
            self.created_features.append('flow_rate')
        
        # Payload size features
        if 'TotLen Fwd Pkts' in df.columns and 'Tot Fwd Pkts' in df.columns:
            df['avg_payload_fwd'] = df['TotLen Fwd Pkts'] / (df['Tot Fwd Pkts'] + 1e-6)
            df['avg_payload_bwd'] = df['TotLen Bwd Pkts'] / (df['Tot Bwd Pkts'] + 1e-6)
            self.created_features.extend(['avg_payload_fwd', 'avg_payload_bwd'])
        
        # Protocol encoding
        if 'Protocol' in df.columns:
            df['protocol_encoded'] = pd.Categorical(df['Protocol']).codes
            self.created_features.append('protocol_encoded')
            
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal attack-specific features"""
        
        # Inter-arrival time features
        if 'Flow IAT Mean' in df.columns:
            df['iat_variance'] = df['Flow IAT Std'] / (df['Flow IAT Mean'] + 1e-6)
            self.created_features.append('iat_variance')
        
        # Burst detection
        if 'Flow Pkts/s' in df.columns:
            mean_pps = df['Flow Pkts/s'].mean()
            df['burst_ratio'] = df['Flow Pkts/s'] / (mean_pps + 1e-6)
            df['is_burst'] = (df['burst_ratio'] > 2.0).astype(int)
            self.created_features.extend(['burst_ratio', 'is_burst'])
        
        # Flag patterns
        flag_cols = ['SYN Flag Cnt', 'RST Flag Cnt', 'ACK Flag Cnt']
        if all(col in df.columns for col in flag_cols):
            df['syn_rst_ratio'] = df['SYN Flag Cnt'] / (df['RST Flag Cnt'] + 1e-6)
            df['unusual_flags'] = ((df['SYN Flag Cnt'] > 0) & (df['RST Flag Cnt'] > 0)).astype(int)
            self.created_features.extend(['syn_rst_ratio', 'unusual_flags'])
            
        return df
    
    def _add_content_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add content attack-specific features"""
        
        # Port analysis
        if 'Dst Port' in df.columns:
            web_ports = [80, 443, 8080, 8443]
            db_ports = [1433, 1521, 3306, 5432]
            
            df['is_web_port'] = df['Dst Port'].isin(web_ports).astype(int)
            df['is_db_port'] = df['Dst Port'].isin(db_ports).astype(int)
            self.created_features.extend(['is_web_port', 'is_db_port'])
        
        # Payload size analysis
        if 'TotLen Fwd Pkts' in df.columns:
            mean_payload = df['TotLen Fwd Pkts'].mean()
            std_payload = df['TotLen Fwd Pkts'].std()
            
            df['unusual_payload'] = (df['TotLen Fwd Pkts'] > mean_payload + 2*std_payload).astype(int)
            df['payload_ratio'] = df['TotLen Fwd Pkts'] / (df['TotLen Bwd Pkts'] + 1e-6)
            self.created_features.extend(['unusual_payload', 'payload_ratio'])
            
        return df
    
    def _add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add behavioral attack-specific features"""
        
        # Connection pattern analysis
        if 'Src Port' in df.columns and 'Dst Port' in df.columns:
            df['is_ephemeral_src'] = (df['Src Port'] > 1024).astype(int)
            df['targets_system_port'] = (df['Dst Port'] < 1024).astype(int)
            df['port_spread'] = abs(df['Src Port'] - df['Dst Port'])
            self.created_features.extend(['is_ephemeral_src', 'targets_system_port', 'port_spread'])
        
        # Session characteristics
        if 'Flow Duration' in df.columns:
            median_duration = df['Flow Duration'].median()
            df['is_short_session'] = (df['Flow Duration'] < median_duration/10).astype(int)
            df['is_long_session'] = (df['Flow Duration'] > median_duration*10).astype(int)
            self.created_features.extend(['is_short_session', 'is_long_session'])
        
        # Volume analysis
        if 'TotLen Fwd Pkts' in df.columns:
            df['is_low_volume'] = ((df['TotLen Fwd Pkts'] < 100) & (df['Tot Fwd Pkts'] < 5)).astype(int)
            self.created_features.append('is_low_volume')
            
        return df

class CentralityFeatureExtractor:
    """Extract community-aware centrality features dynamically"""
    
    def __init__(self):
        self.centrality_cache = {}
    
    def extract_centrality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract centrality features dynamically. 
        If pre-computed columns exist, it uses them. Otherwise, it calculates them.
        """
        import time
        import networkx as nx
        
        logger.info("Calculating centrality features dynamically (Ignoring any dataset pre-computations)...")
        start_time = time.time()
        
        # Drop any existing pre-computed centrality columns to enforce strict local calculation
        existing_centrality = [col for col in df.columns if any(measure in col.lower() for measure in [
            'betweenness', 'pagerank', 'degree', 'closeness', 'eigenvector',
            'k_core', 'k_truss', 'modularity'
        ])]
        
        working_df = df.drop(columns=existing_centrality) if existing_centrality else df.copy()
        
        # We need Src IP and Dst IP. The columns might be 'IPV4_SRC_ADDR' or 'Src IP'
        src_col = 'Src IP' if 'Src IP' in working_df.columns else 'IPV4_SRC_ADDR'
        dst_col = 'Dst IP' if 'Dst IP' in working_df.columns else 'IPV4_DST_ADDR'
        
        if src_col not in working_df.columns or dst_col not in working_df.columns:
            logger.error("Could not find IP columns to build the graph!")
            return working_df
            
        # 1. Build the graph
        # Create edgelist
        edges = list(zip(working_df[src_col], working_df[dst_col]))
        G = nx.Graph()
        G.add_edges_from(edges)
        
        num_nodes = G.number_of_nodes()
        
        # 2. Calculate Fast Metrics
        logger.info(f"Calculating Degree Centrality for {num_nodes} nodes...")
        deg_cent = nx.degree_centrality(G)
        
        logger.info(f"Calculating PageRank...")
        pagerank = nx.pagerank(G)
        
        logger.info(f"Calculating K-Core...")
        core_num = nx.core_number(G)
        
        logger.info(f"Calculating Eigenvector Centrality...")
        try:
            eigen_cent = nx.eigenvector_centrality(G, max_iter=100)
        except Exception as e:
            logger.warning(f"Eigenvector centrality failed to converge: {e}")
            eigen_cent = {node: 0.0 for node in G.nodes()}
            
        # 3. Calculate Approximated Slow Metrics
        k_approx = min(50, num_nodes)
        logger.info(f"Calculating Approximated Betweenness Centrality (k={k_approx})...")
        bet_cent = nx.betweenness_centrality(G, k=k_approx)
        
        # 4. Inject features back into the dataframe
        result_df = working_df.copy()
        
        # Map Src IP
        result_df['src_degree'] = result_df[src_col].map(deg_cent).fillna(0.0)
        result_df['src_pagerank'] = result_df[src_col].map(pagerank).fillna(0.0)
        result_df['src_k_core'] = result_df[src_col].map(core_num).fillna(0.0)
        result_df['src_eigenvector'] = result_df[src_col].map(eigen_cent).fillna(0.0)
        result_df['src_betweenness'] = result_df[src_col].map(bet_cent).fillna(0.0)
        
        # Map Dst IP
        result_df['dst_degree'] = result_df[dst_col].map(deg_cent).fillna(0.0)
        result_df['dst_pagerank'] = result_df[dst_col].map(pagerank).fillna(0.0)
        result_df['dst_k_core'] = result_df[dst_col].map(core_num).fillna(0.0)
        result_df['dst_eigenvector'] = result_df[dst_col].map(eigen_cent).fillna(0.0)
        result_df['dst_betweenness'] = result_df[dst_col].map(bet_cent).fillna(0.0)
        
        elapsed = time.time() - start_time
        logger.info(f"Dynamically calculated 10 centrality features for {len(working_df)} rows in {elapsed:.2f} seconds.")
        
        return result_df
