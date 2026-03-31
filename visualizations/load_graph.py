import igraph as ig

g = ig.Graph.Read_Pickle("../datasets/synthetic/sbm_1000_sparse.pkl")

print(g)