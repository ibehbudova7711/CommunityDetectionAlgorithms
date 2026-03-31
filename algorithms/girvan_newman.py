import igraph as ig
import time

def run_girvan_newman(graph, num_communities=5):
    start_time = time.time()
    
    # Step 1: get dendrogram
    dendrogram = graph.community_edge_betweenness()
    
    # Step 2: cut into desired number of communities
    communities = dendrogram.as_clustering(n=num_communities)
    
    execution_time = time.time() - start_time
    modularity = graph.modularity(communities.membership)
    
    return {
        'algorithm': 'Girvan-Newman',
        'communities': communities.membership,
        'execution_time': execution_time,
        'modularity': modularity,
        'num_communities': len(communities)
    }