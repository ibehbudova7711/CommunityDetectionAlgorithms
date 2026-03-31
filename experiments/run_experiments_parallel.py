import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import igraph as ig
import pandas as pd
import psutil
from multiprocessing import Pool, cpu_count
import os
from algorithms.louvain import run_louvain
from algorithms.spectral import run_spectral
from algorithms.girvan_newman import run_girvan_newman

def test_single_graph(args):
    """
    Test a single graph with all algorithms
    Runs in parallel on separate CPU cores
    """
    filename, size, sparsity = args
    
    print(f"📊 Processing: {size} nodes, {sparsity}")
    
    # Load graph
    graph = ig.Graph.Read_Pickle(filename)
    
    results = []
    
    # Test Louvain
    try:
        louvain_result = run_louvain(graph)

        cpu_usage = psutil.Process().cpu_percent(interval=1)
        print(f"CPU Usage: {cpu_usage}%")

        results.append({
            'graph_size': size,
            'sparsity': sparsity,
            'algorithm': 'Louvain',
            'execution_time': louvain_result['execution_time'],
            'modularity': louvain_result['modularity'],
            'num_communities': louvain_result['num_communities']
            'cpu_percent': cpu_usage
        })
        print(f"   ✅ Louvain done for {size}")
    except Exception as e:
        print(f"   ❌ Louvain failed: {e}")
    
    # Test Spectral
    try:
        spectral_result = run_spectral(graph, k=5)
        cpu_usage = psutil.Process().cpu_percent(interval=1)
        print(f"CPU Usage: {cpu_usage}%")

        results.append({
            'graph_size': size,
            'sparsity': sparsity,
            'algorithm': 'Spectral',
            'execution_time': spectral_result['execution_time'],
            'modularity': spectral_result['modularity'],
            'num_communities': spectral_result['num_communities']
            'cpu_percent': cpu_usage
        })
        print(f"   ✅ Spectral done for {size}")
    except Exception as e:
        print(f"   ❌ Spectral failed: {e}")
    
    # Test Girvan-Newman (only small graphs - it's very slow)
    if size <= 5000:
        try:
            gn_result = run_girvan_newman(graph, num_communities=5)

            cpu_usage = psutil.Process().cpu_percent(interval=1)
            print(f"CPU Usage: {cpu_usage}%")

            results.append({
                'graph_size': size,
                'sparsity': sparsity,
                'algorithm': 'Girvan-Newman',
                'execution_time': gn_result['execution_time'],
                'modularity': gn_result['modularity'],
                'num_communities': gn_result['num_communities']
                'cpu_percent': cpu_usage
            })
            print(f"   ✅ Girvan-Newman done for {size}")
        except Exception as e:
            print(f"   ❌ Girvan-Newman failed: {e}")
    
    return results

def main():
    print("="*60)
    print("PARALLEL COMMUNITY DETECTION EXPERIMENTS")
    print("="*60)
    
    # Check CPU cores
    total_cores = cpu_count()
    print(f"\n💻 Total CPU cores: {total_cores}")
    
    # Use 2-3 workers (leave 1-2 cores for system)
    num_workers = min(3, total_cores - 1)  # For 4 cores, use 3 workers
    print(f"🔧 Using {num_workers} parallel workers")
    print(f"⚡ Expected speedup: ~{num_workers}x faster!\n")
    
    # Prepare all graph tasks
    node_sizes = [1000, 5000, 10000, 20000]
    sparsity_levels = ['very_sparse', 'sparse', 'moderately_sparse']
    
    tasks = []
    for sparsity in sparsity_levels:
        for size in node_sizes:
            filename = f"datasets/synthetic/sbm_{size}_{sparsity}.pkl"
            if os.path.exists(filename):
                tasks.append((filename, size, sparsity))
            else:
                print(f"⚠️  File not found: {filename}")
    
    print(f"📊 Total graphs to test: {len(tasks)}")
    print(f"{'='*60}\n")
    
    # Run experiments in parallel
    all_results = []
    
    print("🚀 Starting parallel experiments...\n")
    
    with Pool(processes=num_workers) as pool:
        # Map tasks to workers (runs in parallel!)
        results_list = pool.map(test_single_graph, tasks)
        
        # Flatten results
        for results in results_list:
            all_results.extend(results)
    
    # Save results
    df = pd.DataFrame(all_results)
    os.makedirs('results', exist_ok=True)
    if os.path.exists('results/experiment_results.csv'):
        df_old = pd.read_csv('results/experiment_results.csv')
        df = pd.concat([df_old, df], ignore_index=True)
    df.to_csv(f'results/experiment_results_parallel.csv', index=False)


    print(f"\n{'='*60}")
    print("✅ ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*60}")
    print(f"📁 Results saved to: results/experiment_results.csv")
    print(f"📊 Total experiments run: {len(all_results)}")
    print(f"\nYou can now run: python visualize.py")

if __name__ == '__main__':
    main()