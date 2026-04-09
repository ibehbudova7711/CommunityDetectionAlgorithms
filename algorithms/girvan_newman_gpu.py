import time
import cudf
import cugraph


def igraph_to_cugraph(graph):
    """
    Convert igraph graph -> cuGraph graph
    """
    edges = graph.get_edgelist()
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]

    df = cudf.DataFrame({
        'src': src,
        'dst': dst
    })

    G = cugraph.Graph()
    G.from_cudf_edgelist(df, source='src', destination='dst', renumber=False)

    return G


def run_gn_gpu(graph, num_communities=5):
    start_time = time.time()

    # Convert graph
    G = igraph_to_cugraph(graph)

    removed_edges = []

    while True:
        # 1. Edge betweenness (GPU)
        eb = cugraph.edge_betweenness_centrality(G)

        # 2. Max edge
        max_row = eb.sort_values(by='edge_betweenness', ascending=False).iloc[0]
        src = int(max_row['src'])
        dst = int(max_row['dst'])

        # 3. Remove edge
        removed_edges.append((src, dst))

        # rebuild graph WITHOUT removed edges
        edges = G.view_edge_list()
        edges = edges[~((edges['src'] == src) & (edges['dst'] == dst))]

        G = cugraph.Graph()
        G.from_cudf_edgelist(edges, source='src', destination='dst', renumber=False)

        # 4. Connected components
        components = cugraph.connected_components(G)
        num_comp = components['labels'].nunique()

        print(f"Current components: {num_comp}", flush=True)

        if num_comp >= num_communities:
            break

    execution_time = time.time() - start_time

    # Convert communities to list
    components = components.to_pandas()
    membership = components.sort_values('vertex')['labels'].tolist()

    return {
        'algorithm': 'Girvan-Newman-GPU',
        'communities': membership,
        'execution_time': execution_time,
        'num_communities': num_comp
    }