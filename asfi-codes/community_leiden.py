import networkx as nx
import igraph as ig
import leidenalg
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def detect_communities_leiden(graph: nx.Graph) -> Dict[int, int]:
    """
    Detect communities using the Leiden algorithm.
    This provides better guarantees than Louvain and is faster.
    """
    logger.info("Detecting communities using Leiden algorithm...")
    
    # Check if graph is empty
    if len(graph) == 0:
        return {}
        
    # Convert NetworkX graph to igraph
    # We must ensure node IDs are mapped correctly since igraph uses contiguous integers 0..N-1
    nx_nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nx_nodes)}
    
    # Create igraph
    g_ig = ig.Graph(directed=False)
    g_ig.add_vertices(len(nx_nodes))
    
    # Add edges with weights
    edges = []
    weights = []
    for u, v, data in graph.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        weights.append(data.get('weight', 1.0))
        
    g_ig.add_edges(edges)
    
    if weights:
        g_ig.es['weight'] = weights
    
    # Run Leiden algorithm
    try:
        partition = leidenalg.find_partition(
            g_ig, 
            leidenalg.ModularityVertexPartition,
            weights='weight' if weights else None
        )
    except Exception as e:
        logger.error(f"Leiden algorithm failed: {e}. Falling back to single community.")
        return {node: 0 for node in graph.nodes()}
        
    # Convert partition back to dict {original_node: community_id}
    communities = {}
    for idx, community_id in enumerate(partition.membership):
        original_node = nx_nodes[idx]
        communities[original_node] = community_id
        
    logger.info(f"Detected {len(set(communities.values()))} communities using Leiden")
    return communities
