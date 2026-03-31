import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import igraph as ig
import pandas as pd
from memory_profiler import memory_usage
import psutil
from algorithms.louvain import run_louvain
from algorithms.spectral import run_spectral
from algorithms.girvan_newman import run_girvan_newman

def save_results(results):
    df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/experiment_results.csv', index=False)

print("="*60)
print("COMMUNITY DETECTION EXPERIMENTS")
print("="*60)

# Results storage
if os.path.exists('results/experiment_results.csv'):
    df_old = pd.read_csv('results/experiment_results.csv')
    results = df_old.to_dict('records')
else:
    results = []


# Test on synthetic graphs
node_sizes = [1000, 5000, 10000, 20000]
sparsity_levels = ['very_sparse', 'sparse', 'moderately_sparse']

for sparsity in sparsity_levels:
    for size in node_sizes:
        filename = f"datasets/synthetic/sbm_{size}_{sparsity}.pkl"
        
        if not os.path.exists(filename):
            print(f"⚠️  Skipping {filename} (not found)")
            continue
            
        print(f"\n{'='*60}")
        print(f"Testing: {size} nodes, {sparsity}")
        print(f"{'='*60}")
        
        # Load graph
        graph = ig.Graph.Read_Pickle(filename)
        
        # Test Louvain
        print("Running Louvain...")
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        louvain_result = run_louvain(graph)
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024
        mem_used = mem_after - mem_before
        
        results.append({
            'graph_type': 'synthetic',
            'graph_size': size,
            'sparsity': sparsity,
            'algorithm': 'Louvain',
            'execution_time': louvain_result['execution_time'],
            'memory_mb': mem_used,
            'modularity': louvain_result['modularity'],
            'num_communities': louvain_result['num_communities']
        })
        save_results(results)

        print(f"   Time: {louvain_result['execution_time']:.4f}s")
        print(f"   Modularity: {louvain_result['modularity']:.4f}")

        
        # Test Spectral
        print("Running Spectral...")
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024
        spectral_result = run_spectral(graph, k=5)
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024
        mem_used = mem_after - mem_before
        
        results.append({
            'graph_type': 'synthetic',
            'graph_size': size,
            'sparsity': sparsity,
            'algorithm': 'Spectral',
            'execution_time': spectral_result['execution_time'],
            'memory_mb': mem_used,
            'modularity': spectral_result['modularity'],
            'num_communities': spectral_result['num_communities']
        })
        save_results(results)

        print(f"   Time: {spectral_result['execution_time']:.4f}s")
        print(f"   Modularity: {spectral_result['modularity']:.4f}")
        
        # Test Girvan-Newman (only for smaller graphs - it's very slow!)
        if size <= 1000:  # Skip for large graphs
            print("Running Girvan-Newman...")
            mem_before = psutil.Process().memory_info().rss / 1024 / 1024
            gn_result = run_girvan_newman(graph, num_communities=5)
            mem_after = psutil.Process().memory_info().rss / 1024 / 1024
            mem_used = mem_after - mem_before
            
            results.append({
                'graph_type': 'synthetic',
                'graph_size': size,
                'sparsity': sparsity,
                'algorithm': 'Girvan-Newman',
                'execution_time': gn_result['execution_time'],
                'memory_mb': mem_used,
                'modularity': gn_result['modularity'],
                'num_communities': gn_result['num_communities']
            })

            save_results(results)
            print(f"   Time: {gn_result['execution_time']:.4f}s")
            print(f"   Modularity: {gn_result['modularity']:.4f}")
        else:
            print("   Skipping Girvan-Newman (too slow for large graphs)")

# Test on Facebook graph
print(f"\n{'='*60}")
print("Testing on Facebook dataset")
print(f"{'='*60}")

facebook_graph = ig.Graph.Read_Pickle('datasets/facebook/facebook_graph.pkl')

# Louvain on Facebook
print("Running Louvain on Facebook...")
louvain_result = run_louvain(facebook_graph)
results.append({
    'graph_type': 'facebook',
    'graph_size': facebook_graph.vcount(),
    'sparsity': 'real_world',
    'algorithm': 'Louvain',
    'execution_time': louvain_result['execution_time'],
    'memory_mb': 0,  # Not measured for real data
    'modularity': louvain_result['modularity'],
    'num_communities': louvain_result['num_communities']
})
save_results(results)

# Spectral on Facebook

# k is a hyperparameter for real data (no ground truth)
# tested with k=10 based on experimentation
print("Running Spectral on Facebook...")
spectral_result = run_spectral(facebook_graph, k=10)
results.append({
    'graph_type': 'facebook',
    'graph_size': facebook_graph.vcount(),
    'sparsity': 'real_world',
    'algorithm': 'Spectral',
    'execution_time': spectral_result['execution_time'],
    'memory_mb': 0,
    'modularity': spectral_result['modularity'],
    'num_communities': spectral_result['num_communities']
})
save_results(results)

# Save results
df = pd.DataFrame(results)
os.makedirs('results', exist_ok=True)
df.to_csv('results/experiment_results.csv', index=False)

print(f"\n{'='*60}")
print("✅ ALL EXPERIMENTS COMPLETED!")
print(f"{'='*60}")
print(f"Results saved to: results/experiment_results.csv")
print(f"Total experiments: {len(results)}")
print("\nSummary:")
print(df.groupby('algorithm')['modularity'].mean())