# IMPORTANT:
# For synthetic graphs → k=5 (known ground truth)
# For real graphs → experiment with different k values:
# run_spectral(graph, k=3)
# run_spectral(graph, k=5)
# run_spectral(graph, k=10)
# Then compare modularity and choose the best

import igraph as ig
import time

def run_spectral(graph, k=5):
    start_time = time.time()
    
    communities = graph.community_leading_eigenvector(clusters=k)
    
    execution_time = time.time() - start_time
    modularity = graph.modularity(communities.membership)
    
    return {
        'algorithm': 'Spectral',
        'communities': communities.membership,
        'execution_time': execution_time,
        'modularity': modularity,
        'num_communities': len(communities)
    }