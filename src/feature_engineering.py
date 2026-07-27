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
        
        # Flow rate features (CIC format)
        if 'Flow Duration' in df.columns and 'Tot Fwd Pkts' in df.columns:
            df['flow_rate'] = (df['Tot Fwd Pkts'] + df['Tot Bwd Pkts']) / (df['Flow Duration'] / 1000000 + 1e-6)
            self.created_features.append('flow_rate')
        
        # Payload size features (CIC format)
        if 'TotLen Fwd Pkts' in df.columns and 'Tot Fwd Pkts' in df.columns:
            df['avg_payload_fwd'] = df['TotLen Fwd Pkts'] / (df['Tot Fwd Pkts'] + 1e-6)
            df['avg_payload_bwd'] = df['TotLen Bwd Pkts'] / (df['Tot Bwd Pkts'] + 1e-6)
            self.created_features.extend(['avg_payload_fwd', 'avg_payload_bwd'])
        
        # Protocol encoding (CIC format)
        if 'Protocol' in df.columns:
            df['protocol_encoded'] = pd.Categorical(df['Protocol']).codes
            self.created_features.append('protocol_encoded')
        
        # --- NF-format discriminative features (Option A) ---
        # These features specifically target the DDoS/Injection/Password/XSS confusion problem.
        
        # 1. out_in_bytes_ratio: Injection returns large server responses (DB dumps/errors),
        #    DDoS sends tiny requests and gets small/no response.
        #    Key discriminator: injection >> password > xss >> ddos
        if 'IN_BYTES' in df.columns and 'OUT_BYTES' in df.columns:
            df['out_in_bytes_ratio'] = (df['OUT_BYTES'] / (df['IN_BYTES'] + 1e-6)).clip(0, 100).fillna(0)
            self.created_features.append('out_in_bytes_ratio')
        
        # 2. bytes_per_packet: DDoS floods with tiny packets (low value).
        #    Injection/password involve real TCP sessions with larger payloads (high value).
        if 'IN_BYTES' in df.columns and 'IN_PKTS' in df.columns and 'OUT_BYTES' in df.columns and 'OUT_PKTS' in df.columns:
            total_bytes = df['IN_BYTES'] + df['OUT_BYTES']
            total_pkts = df['IN_PKTS'] + df['OUT_PKTS']
            df['bytes_per_packet'] = (total_bytes / (total_pkts + 1e-6)).clip(0, 10000).fillna(0)
            self.created_features.append('bytes_per_packet')
        
        # 3. pkt_asymmetry: measures directionality of the flow.
        #    DDoS is typically one-directional (high asymmetry).
        #    Injection/password are bidirectional (low asymmetry, server sends back data).
        if 'IN_PKTS' in df.columns and 'OUT_PKTS' in df.columns:
            total_pkts = df['IN_PKTS'] + df['OUT_PKTS']
            df['pkt_asymmetry'] = (abs(df['IN_PKTS'] - df['OUT_PKTS']) / (total_pkts + 1e-6)).clip(0, 1).fillna(0)
            self.created_features.append('pkt_asymmetry')
        
        # 4. NF flow rate (NF-format equivalent of CIC flow_rate)
        #    Clipped to 1000 pkts/ms to guard against near-zero duration DDoS rows.
        if 'FLOW_DURATION_MILLISECONDS' in df.columns and 'IN_PKTS' in df.columns:
            total_pkts = df['IN_PKTS'] + df.get('OUT_PKTS', 0)
            df['nf_flow_rate'] = (total_pkts / (df['FLOW_DURATION_MILLISECONDS'] + 1e-3)).clip(0, 1000).fillna(0)
            self.created_features.append('nf_flow_rate')

        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal attack-specific features"""
        
        # Inter-arrival time features (CIC format)
        if 'Flow IAT Mean' in df.columns:
            df['iat_variance'] = df['Flow IAT Std'] / (df['Flow IAT Mean'] + 1e-6)
            self.created_features.append('iat_variance')
        
        # Burst detection (CIC format)
        if 'Flow Pkts/s' in df.columns:
            mean_pps = df['Flow Pkts/s'].mean()
            df['burst_ratio'] = df['Flow Pkts/s'] / (mean_pps + 1e-6)
            df['is_burst'] = (df['burst_ratio'] > 2.0).astype(int)
            self.created_features.extend(['burst_ratio', 'is_burst'])
        
        # Flag patterns (CIC format)
        flag_cols = ['SYN Flag Cnt', 'RST Flag Cnt', 'ACK Flag Cnt']
        if all(col in df.columns for col in flag_cols):
            df['syn_rst_ratio'] = df['SYN Flag Cnt'] / (df['RST Flag Cnt'] + 1e-6)
            df['unusual_flags'] = ((df['SYN Flag Cnt'] > 0) & (df['RST Flag Cnt'] > 0)).astype(int)
            self.created_features.extend(['syn_rst_ratio', 'unusual_flags'])
        
        # --- Option A: DDoS discriminator via flows_per_src_ip ---
        # DDoS attackers generate a huge number of flows from the same source IP.
        # Counting how many times each src IP appears in this batch is a strong signal.
        src_col = 'IPV4_SRC_ADDR' if 'IPV4_SRC_ADDR' in df.columns else ('Src IP' if 'Src IP' in df.columns else None)
        dst_ip_col = 'IPV4_DST_ADDR' if 'IPV4_DST_ADDR' in df.columns else ('Dst IP' if 'Dst IP' in df.columns else None)
        dst_port_col = 'L4_DST_PORT' if 'L4_DST_PORT' in df.columns else ('Dst Port' if 'Dst Port' in df.columns else None)

        if src_col:
            flows_per_src = df[src_col].map(df[src_col].value_counts()).fillna(1)
            df['flows_per_src_ip'] = flows_per_src
            # Normalise to fraction of the batch so value scales with batch size
            df['flows_per_src_ip_norm'] = (flows_per_src / (len(df) + 1e-6)).clip(0, 1).fillna(0)
            self.created_features.extend(['flows_per_src_ip', 'flows_per_src_ip_norm'])

        # --- Scanning vs Password discriminators via destination diversity ---
        # Scanning: one src IP connects to MANY different dst IPs and dst ports (high diversity)
        # Password: one src IP repeatedly hits the SAME dst IP + port (low diversity)
        # Injection: one src IP hits the same web server (low-medium diversity)
        if src_col and dst_ip_col:
            # Unique dst IPs per src IP in this batch
            dst_ip_diversity = df.groupby(src_col)[dst_ip_col].transform('nunique').fillna(1)
            df['dst_ip_diversity'] = dst_ip_diversity.clip(1, None)
            # Log-scale to compress the range (scanning can have 100+ unique IPs)
            df['dst_ip_diversity_log'] = np.log1p(dst_ip_diversity).fillna(0)
            self.created_features.extend(['dst_ip_diversity', 'dst_ip_diversity_log'])

        if src_col and dst_port_col:
            # Unique dst ports per src IP in this batch
            dst_port_diversity = df.groupby(src_col)[dst_port_col].transform('nunique').fillna(1)
            df['dst_port_diversity'] = dst_port_diversity.clip(1, None)
            df['dst_port_diversity_log'] = np.log1p(dst_port_diversity).fillna(0)
            self.created_features.extend(['dst_port_diversity', 'dst_port_diversity_log'])

            # High port diversity ratio: scanning has high port spread relative to flow count
            if src_col in df.columns:
                df['port_to_flow_ratio'] = (dst_port_diversity / (flows_per_src + 1e-6)).clip(0, 1).fillna(0)
                self.created_features.append('port_to_flow_ratio')

        return df
    
    def _add_content_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add content attack-specific features"""
        
        # Port analysis (CIC format)
        if 'Dst Port' in df.columns:
            web_ports = [80, 443, 8080, 8443]
            db_ports = [1433, 1521, 3306, 5432]
            
            df['is_web_port'] = df['Dst Port'].isin(web_ports).astype(int)
            df['is_db_port'] = df['Dst Port'].isin(db_ports).astype(int)
            self.created_features.extend(['is_web_port', 'is_db_port'])
        
        # NF-format port analysis
        if 'L4_DST_PORT' in df.columns:
            web_ports = [80, 443, 8080, 8443]
            db_ports = [1433, 1521, 3306, 5432]
            df['is_web_port'] = df['L4_DST_PORT'].isin(web_ports).astype(int)
            df['is_db_port'] = df['L4_DST_PORT'].isin(db_ports).astype(int)
            self.created_features.extend(['is_web_port', 'is_db_port'])
        
        # Payload size analysis (CIC format)
        if 'TotLen Fwd Pkts' in df.columns:
            mean_payload = df['TotLen Fwd Pkts'].mean()
            std_payload = df['TotLen Fwd Pkts'].std()
            
            df['unusual_payload'] = (df['TotLen Fwd Pkts'] > mean_payload + 2*std_payload).astype(int)
            df['payload_ratio'] = df['TotLen Fwd Pkts'] / (df['TotLen Bwd Pkts'] + 1e-6)
            self.created_features.extend(['unusual_payload', 'payload_ratio'])
        
        # --- Option A: Injection vs XSS discriminator via response_size_category ---
        # Injection attacks trigger large server responses (DB error messages, data dumps).
        # XSS attacks typically get small confirmation responses.
        # The out_in_bytes_ratio from base features is key; we add a categorical bin here.
        if 'OUT_BYTES' in df.columns and 'IN_BYTES' in df.columns:
            ratio = (df['OUT_BYTES'] / (df['IN_BYTES'] + 1e-6)).clip(0, 100)
            # Bin: 0=minimal(xss-like), 1=balanced, 2=large-response(injection-like)
            df['response_size_category'] = pd.cut(
                ratio, bins=[-1, 1.0, 3.0, float('inf')], labels=[0, 1, 2]
            ).astype(float).fillna(0)
            self.created_features.append('response_size_category')
            
        # --- Password vs Injection discriminator features for Content GAT ---
        # Password brute-force login attempts have distinct outbound byte density and flow duration
        # compared to complex server-evaluated Injection payloads.
        if 'OUT_BYTES' in df.columns and 'OUT_PKTS' in df.columns:
            df['byte_per_pkt_out'] = (df['OUT_BYTES'] / (df['OUT_PKTS'] + 1e-6)).clip(0, 10000).fillna(0)
            self.created_features.append('byte_per_pkt_out')
        if 'OUT_PKTS' in df.columns and 'IN_PKTS' in df.columns:
            df['out_in_pkts_ratio'] = (df['OUT_PKTS'] / (df['IN_PKTS'] + 1e-6)).clip(0, 50).fillna(0)
            self.created_features.append('out_in_pkts_ratio')
        if 'IN_BYTES' in df.columns and 'OUT_BYTES' in df.columns and 'FLOW_DURATION_MILLISECONDS' in df.columns:
            df['byte_density'] = ((df['IN_BYTES'] + df['OUT_BYTES']) / (df['FLOW_DURATION_MILLISECONDS'] + 1e-3)).clip(0, 100000).fillna(0)
            self.created_features.append('byte_density')
            
        return df
    
    def _add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add behavioral attack-specific features"""
        
        # Connection pattern analysis (CIC format)
        if 'Src Port' in df.columns and 'Dst Port' in df.columns:
            df['is_ephemeral_src'] = (df['Src Port'] > 1024).astype(int)
            df['targets_system_port'] = (df['Dst Port'] < 1024).astype(int)
            df['port_spread'] = abs(df['Src Port'] - df['Dst Port'])
            self.created_features.extend(['is_ephemeral_src', 'targets_system_port', 'port_spread'])
        
        # NF-format connection pattern analysis
        if 'L4_SRC_PORT' in df.columns and 'L4_DST_PORT' in df.columns:
            df['is_ephemeral_src'] = (df['L4_SRC_PORT'] > 1024).astype(int)
            df['targets_system_port'] = (df['L4_DST_PORT'] < 1024).astype(int)
            df['port_spread'] = abs(df['L4_SRC_PORT'] - df['L4_DST_PORT'])
            self.created_features.extend(['is_ephemeral_src', 'targets_system_port', 'port_spread'])
        
        # Session characteristics (CIC format)
        if 'Flow Duration' in df.columns:
            median_duration = df['Flow Duration'].median()
            df['is_short_session'] = (df['Flow Duration'] < median_duration/10).astype(int)
            df['is_long_session'] = (df['Flow Duration'] > median_duration*10).astype(int)
            self.created_features.extend(['is_short_session', 'is_long_session'])
        
        # Session characteristics (NF format)
        if 'FLOW_DURATION_MILLISECONDS' in df.columns:
            median_duration = df['FLOW_DURATION_MILLISECONDS'].median()
            df['is_short_session'] = (df['FLOW_DURATION_MILLISECONDS'] < median_duration/10).astype(int)
            df['is_long_session'] = (df['FLOW_DURATION_MILLISECONDS'] > median_duration*10).astype(int)
            self.created_features.extend(['is_short_session', 'is_long_session'])
        
        # Volume analysis (CIC format)
        if 'TotLen Fwd Pkts' in df.columns:
            df['is_low_volume'] = ((df['TotLen Fwd Pkts'] < 100) & (df['Tot Fwd Pkts'] < 5)).astype(int)
            self.created_features.append('is_low_volume')
        
        # --- Option A: Password brute-force discriminator via session_regularity ---
        # Password attacks repeat the same connection pattern (same port, same bytes) many times.
        # We measure this by checking how uniform the IN_BYTES values are in this batch.
        # High uniformity = likely brute-force (password), low uniformity = legitimate or XSS.
        if 'IN_BYTES' in df.columns:
            dst_col = 'L4_DST_PORT' if 'L4_DST_PORT' in df.columns else ('Dst Port' if 'Dst Port' in df.columns else None)
            if dst_col:
                # Coefficient of variation of IN_BYTES per destination port group
                # Password brute-force always sends same-sized requests → low CV
                port_cv = df.groupby(dst_col)['IN_BYTES'].transform(
                    lambda x: x.std() / (x.mean() + 1e-6)
                ).fillna(0)  # fillna: groups with 1 member return NaN std()
                df['session_regularity'] = (1.0 / (port_cv + 1e-6)).clip(0, 100).fillna(0)
                self.created_features.append('session_regularity')
            
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
        G.remove_edges_from(nx.selfloop_edges(G))
        
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
