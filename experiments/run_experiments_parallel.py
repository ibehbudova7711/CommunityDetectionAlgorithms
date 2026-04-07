import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import igraph as ig
import pandas as pd
import psutil
from multiprocessing import Pool, cpu_count

from algorithms.louvain import run_louvain
from algorithms.spectral import run_spectral
from algorithms.girvan_newman import run_girvan_newman

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# =========================
# SAVE FUNCTION
# =========================
def save_results(results):
    df = pd.DataFrame(results)

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    file_path = os.path.join(results_dir, "experiment_results_parallel.csv")

    if os.path.exists(file_path):
        df_old = pd.read_csv(file_path)
        df = pd.concat([df_old, df], ignore_index=True)

    df.to_csv(file_path, index=False)


# =========================
# CHECK DONE
# =========================
def already_done(size, sparsity, graph_type):
    file_path = os.path.join(BASE_DIR, "results", "experiment_results_parallel.csv")

    if not os.path.exists(file_path):
        return False

    df = pd.read_csv(file_path)

    return ((df['graph_size'] == size) &
            (df['sparsity'] == sparsity) &
            (df['graph_type'] == graph_type) &
            (df['algorithm'] == 'Girvan-Newman')).any()


# =========================
# FAST PHASE (UNCHANGED)
# =========================
def run_fast_algorithms(args):
    filename, size, sparsity, graph_type = args

    print(f"\n FAST: {size} | {sparsity} | {graph_type}", flush=True)

    graph = ig.Graph.Read_Pickle(filename)
    results = []

    try:
        res = run_louvain(graph)
        results.append({
            'graph_type': graph_type,
            'graph_size': size,
            'sparsity': sparsity,
            'algorithm': 'Louvain',
            'execution_time': res['execution_time'],
            'modularity': res['modularity'],
            'num_communities': res['num_communities'],
            'cpu_percent': psutil.cpu_percent(interval=None)
        })
        print(f"   Louvain done ({size}, {sparsity}, {graph_type})", flush=True)
    except Exception as e:
        print(f" Louvain error: {e}", flush=True)

    try:
        k = 10 if graph_type == 'facebook' else 5
        res = run_spectral(graph, k=k)
        results.append({
            'graph_type': graph_type,
            'graph_size': size,
            'sparsity': sparsity,
            'algorithm': 'Spectral',
            'execution_time': res['execution_time'],
            'modularity': res['modularity'],
            'num_communities': res['num_communities'],
            'cpu_percent': psutil.cpu_percent(interval=None)
        })
        print(f"  Spectral done ({size}, {sparsity}, {graph_type})", flush=True)
    except Exception as e:
        print(f" Spectral error: {e}", flush=True)

    return results


# =========================
# GN PARALLEL WORKER
# =========================
def run_gn_single(args):
    filename, size, sparsity, graph_type = args

    if already_done(size, sparsity, graph_type):
        print(f" Skipping already done: {size} | {sparsity} | {graph_type}", flush=True)
        return

    print(f"\n GN START: {size} | {sparsity} | {graph_type}", flush=True)

    graph = ig.Graph.Read_Pickle(filename)

    try:
        print(f" Running GN... may take long ({size})", flush=True)

        res = run_girvan_newman(graph, num_communities=5)

        result = {
            'graph_type': graph_type,
            'graph_size': size,
            'sparsity': sparsity,
            'algorithm': 'Girvan-Newman',
            'execution_time': res['execution_time'],
            'modularity': res['modularity'],
            'num_communities': res['num_communities'],
            'cpu_percent': psutil.cpu_percent(interval=None)
        }

        save_results([result])

        print(f"    GN done ({size}, {sparsity}, {graph_type})", flush=True)

    except Exception as e:
        print(f" GN error ({size}, {sparsity}, {graph_type}): {e}", flush=True)


# =========================
# MAIN
# =========================
def main():
    print("🚀 STARTING EXPERIMENTS")

    node_sizes = [1000, 5000, 10000, 20000]
    sparsity_levels = ['very_sparse', 'sparse', 'moderately_sparse']

    tasks = []

    # Synthetic
    for sparsity in sparsity_levels:
        for size in node_sizes:
            path = os.path.join(BASE_DIR, f"datasets/synthetic/sbm_{size}_{sparsity}.pkl")
            if os.path.exists(path):
                tasks.append((path, size, sparsity, 'synthetic'))

    # Facebook
    fb_path = os.path.join(BASE_DIR, "datasets/facebook_graph.pkl")
    if os.path.exists(fb_path):
        tasks.append((fb_path, 4039, 'real_world', 'facebook'))

    print(f" Total tasks: {len(tasks)}")

    # FAST phase (unchanged)
    print("\n RUNNING FAST ALGORITHMS...\n")

    with Pool(processes=cpu_count()) as pool:
        fast_results_list = pool.map(run_fast_algorithms, tasks)

    fast_results = [item for sublist in fast_results_list for item in sublist]
    save_results(fast_results)

    print("\n FAST PHASE DONE")

    # GN phase (NOW PARALLEL)
    print("\n STARTING GN PHASE...\n")

    with Pool(processes=cpu_count()) as pool:
        pool.map(run_gn_single, tasks)

    print("\n ALL EXPERIMENTS FINISHED")


if __name__ == "__main__":
    main()