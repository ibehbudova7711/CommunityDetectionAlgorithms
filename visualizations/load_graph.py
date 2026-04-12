import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/experiment_results_parallel.csv")

df["modularity_scaled"] = df["modularity"] * 1000
metrics = ["execution_time", "modularity_scaled", "memory_usage"]

# =======================
# CPU vs GPU SCATTER PLOT
# =======================

# Load data
cpu_df = pd.read_csv("results/experiment_results_parallel.csv")
gpu_df = pd.read_csv("results/gn_gpu_results.csv")

# Clean columns
cpu_df.columns = cpu_df.columns.str.strip()
gpu_df.columns = gpu_df.columns.str.strip()

cpu_df["algorithm"] = cpu_df["algorithm"].str.strip()
gpu_df["algorithm"] = gpu_df["algorithm"].str.strip()

# Filter Girvan-Newman
cpu_df = cpu_df[cpu_df["algorithm"] == "Girvan-Newman"]
gpu_df = gpu_df[gpu_df["algorithm"] == "Girvan-Newman-GPU"]

# Merge
merged = pd.merge(
    cpu_df,
    gpu_df,
    on=["graph_size", "sparsity"],
    suffixes=("_cpu", "_gpu")
)

# Order sparsity
order = ["very_sparse", "sparse", "moderately_sparse"]
merged["sparsity"] = pd.Categorical(merged["sparsity"], categories=order, ordered=True)

# Group and reindex 
avg = merged.groupby("sparsity").mean(numeric_only=True).reindex(order)

# AVERAGE over graph sizes (important!)
avg = merged.groupby("sparsity").mean(numeric_only=True)

# Plot
plt.figure(figsize=(8,5))

# CPU line
plt.plot(
    avg.index,
    avg["execution_time_cpu"],
    marker='o',
    label="CPU"
)

# GPU line
plt.plot(
    avg.index,
    avg["execution_time_gpu"],
    marker='o',
    label="GPU"
)

plt.xlabel("Sparsity Level")
plt.ylabel("Execution Time")
plt.title("CPU vs GPU Execution Time across Sparsity Levels")

plt.legend()
plt.grid()

plt.show()

# =======================
# MODULARITY vs SPARSITY (FIXED)
# =======================

plt.figure(figsize=(8,5))

order = ["very_sparse", "sparse", "moderately_sparse"]

grouped_sparse = df.groupby(["sparsity", "algorithm"]).mean(numeric_only=True).reset_index()

# 🔥 KRİTİK CLEANING
grouped_sparse = grouped_sparse.dropna(subset=["sparsity"])

# yalnız düzgün dəyərləri saxla
grouped_sparse = grouped_sparse[grouped_sparse["sparsity"].isin(order)]

# string-ə çevir
grouped_sparse["sparsity"] = grouped_sparse["sparsity"].astype(str)

for algo in grouped_sparse["algorithm"].unique():
    subset = grouped_sparse[grouped_sparse["algorithm"] == algo].copy()
    
    subset["sparsity"] = pd.Categorical(subset["sparsity"], categories=order, ordered=True)
    subset = subset.sort_values("sparsity")
    
    plt.plot(
        subset["sparsity"].astype(str),   # 🔥 BURASI vacibdir
        subset["modularity"],
        marker='o',
        label=algo
    )

plt.xlabel("Sparsity Level")
plt.ylabel("Modularity")
plt.title("Modularity vs Sparsity (CPU Algorithms)")

plt.legend()
plt.grid()

plt.show()

# Clean columns
df.columns = df.columns.str.strip()
df["algorithm"] = df["algorithm"].str.strip()

# Only synthetic graphs (optional)
#df = df[df["graph_type"] == "synthetic"]

# Group data
grouped = df.groupby(["graph_size", "algorithm"]).mean(numeric_only=True).reset_index()

# ========= 1. EXECUTION TIME =========
plt.figure(figsize=(8,5))

for algo in grouped["algorithm"].unique():
    subset = grouped[grouped["algorithm"] == algo]
    subset = subset.sort_values("graph_size")
    
    plt.plot(
        subset["graph_size"],
        subset["execution_time"],
        marker='o',
        label=algo
    )

plt.xlabel("Graph Size")
plt.ylabel("Execution Time")
plt.title("Execution Time vs Graph Size")
plt.yscale("log")

plt.legend()
plt.grid()
plt.show()


# ========= 2. MODULARITY =========
plt.figure(figsize=(8,5))

for algo in grouped["algorithm"].unique():
    subset = grouped[grouped["algorithm"] == algo]
    
    plt.plot(
        subset["graph_size"],
        subset["modularity"],
        marker='o',
        label=algo
    )

plt.xlabel("Graph Size")
plt.ylabel("Modularity")
plt.title("Modularity vs Graph Size")

plt.legend()
plt.grid()
plt.show()


# ========= 3. MEMORY USAGE =========
plt.figure(figsize=(8,5))

for algo in grouped["algorithm"].unique():
    subset = grouped[grouped["algorithm"] == algo]
    
    plt.plot(
        subset["graph_size"],
        subset["memory_usage"],
        marker='o',
        label=algo
    )

plt.xlabel("Graph Size")
plt.ylabel("Memory Usage")
plt.title("Memory Usage vs Graph Size")

plt.legend()
plt.grid()
plt.show()