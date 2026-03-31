import igraph as ig
import time
def run_louvain(graph):
    start_time = time.time()
    
    communities = graph.community_multilevel()
    
    execution_time = time.time() - start_time
    modularity = graph.modularity(communities.membership)
    
    return {
        'algorithm': 'Louvain',
        'communities': communities.membership,
        'execution_time': execution_time,
        'modularity': modularity,
        'num_communities': len(communities)
    }
