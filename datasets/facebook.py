import igraph as ig

print("Loading Facebook dataset from TXT...")

# Load edges from facebook.txt
edges = []
with open('facebook.txt', 'r') as f:
    for line in f:
        # Skip empty lines or comments
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Parse edge
        parts = line.split()
        node1, node2 = int(parts[0]), int(parts[1])
        edges.append((node1, node2))

# Create graph
facebook_graph = ig.Graph(edges)

# Save as pickle (much faster to load later!)
facebook_graph.write_pickle('facebook_graph.pkl')

print(f"Facebook graph converted to PKL!")
print(f"Nodes: {facebook_graph.vcount()}")
print(f"Edges: {facebook_graph.ecount()}")
print(f"Saved to: facebook_graph.pkl")