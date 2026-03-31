import igraph as ig
import os
import json
 
os.makedirs("synthetic", exist_ok=True)

print("="*60)
print("LARGE SPARSE GRAPH GENERATOR")
print("="*60)

node_sizes = [1000, 5000, 10000, 20000]
num_communities = 5

sparsity_configs = {
    'very_sparse': {'within': 0.05, 'between': 0.001},
    'sparse': {'within': 0.1, 'between': 0.005},
    'moderately_sparse': {'within': 0.2, 'between': 0.01}
}

all_stats = []

for sparsity_name, probs in sparsity_configs.items():
    print(f"\n{'='*60}")
    print(f"Sparsity Level: {sparsity_name.upper()}")
    print(f"{'='*60}")
    
    within_prob = probs['within']
    between_prob = probs['between']
    
    for n in node_sizes:
        print(f"\n📊 Generating: {n:,} nodes, {sparsity_name}")
        
        base_size = n // num_communities
        block_sizes = [base_size] * num_communities
        
        remainder = n % num_communities
        for i in range(remainder):
            block_sizes[i] += 1
        
        print(f"   Communities: {num_communities}")
        print(f"   Community sizes: {block_sizes}")
        
        pref_matrix = []
        for i in range(num_communities):
            row = []
            for j in range(num_communities):
                if i == j:
                    row.append(within_prob)
                else:
                    row.append(between_prob)
            pref_matrix.append(row)
        
        print(f"   Within-community prob: {within_prob}")
        print(f"   Between-community prob: {between_prob}")
        
        try:
            graph = ig.Graph.SBM(
                pref_matrix=pref_matrix,
                block_sizes=block_sizes,
                directed=False
            )
            
            num_nodes = graph.vcount()
            num_edges = graph.ecount()
            density = graph.density()
            avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0
            
            max_edges = (num_nodes * (num_nodes - 1)) / 2
            sparsity_ratio = (num_edges / max_edges) * 100
            
            print(f"\n   ✅ CREATED:")
            print(f"   Nodes: {num_nodes:,}")
            print(f"   Edges: {num_edges:,}")
            print(f"   Density: {density:.6f}")
            print(f"   Sparsity: {100-sparsity_ratio:.2f}%")
            print(f"   Avg Degree: {avg_degree:.2f}")
            
            # ✅ PATH FIX
            filename = f"sbm_{n}_{sparsity_name}.pkl"
            filepath = f"synthetic/{filename}"
            graph.write_pickle(filepath)
            print(f"   💾 Saved: {filename}")
            
            # ✅ PATH FIX
            edge_file = f"synthetic/sbm_{n}_{sparsity_name}_edges.txt"
            with open(edge_file, 'w') as f:
                for edge in graph.get_edgelist():
                    f.write(f"{edge[0]} {edge[1]}\n")
            
            all_stats.append({
                'nodes': num_nodes,
                'edges': num_edges,
                'density': density,
                'sparsity_percent': 100 - sparsity_ratio,
                'avg_degree': avg_degree,
                'communities': num_communities,
                'sparsity_level': sparsity_name,
                'within_prob': within_prob,
                'between_prob': between_prob
            })
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            continue

# ✅ JSON SAVE (artıq düzgün idi)
stats_file = 'synthetic/sparse_graph_stats.json'
with open(stats_file, 'w') as f:
    json.dump(all_stats, f, indent=2)

print(f"\n{'='*60}")
print("✅ ALL LARGE SPARSE GRAPHS GENERATED!")
print(f"{'='*60}")
print(f"Total graphs created: {len(all_stats)}")
print(f"Statistics saved to: {stats_file}")