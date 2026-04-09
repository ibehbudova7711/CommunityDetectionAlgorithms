import os
import igraph as ig
import pandas as pd
import psutil
from multiprocessing import Pool

from algorithms.girvan_newman_gpu import run_gn_gpu

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def save_results(results):
    df = pd.DataFrame(results)

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    file_path = os.path.join(results_dir, "gn_gpu_results.csv")

    if os.path.exists(file_path):
        df_old = pd.read_csv(file_path)
        df = pd.concat([df_old, df], ignore_index=True)

    df.to_csv(file_path, index=False)


def run_single(task):
    filename, size, sparsity, graph_type = task

    print(f"\n🚀 GN GPU START: {size} | {sparsity} | {graph_type}", flush=True)

    graph = ig.Graph.Read_Pickle(filename)

    try:
        res = run_gn_gpu(graph, num_communities=5)

        result = {
            'graph_type': graph_type,
            'graph_size': size,
            'sparsity': sparsity,
            'algorithm': 'Girvan-Newman-GPU',
            'execution_time': res['execution_time'],
            'num_communities': res['num_communities'],
            'cpu_percent': psutil.cpu_percent(interval=None)
        }

        save_results([result])

        print(f" DONE: {size} | {sparsity} | {graph_type}", flush=True)

    except Exception as e:
        print(f" ERROR: {e}", flush=True)


def main():
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

    print(f"Total tasks: {len(tasks)}")

    # IMPORTANT: limit processes (GPU contention)
    with Pool(processes=2) as pool:
        pool.map(run_single, tasks)


if __name__ == "__main__":
    main()