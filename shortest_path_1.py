import networkx as nx
import matplotlib.pyplot as plt

# Create a random graph
G = nx.gnp_random_graph(10, 0.5)

# Calculate the shortest path
path = nx.shortest_path(G, source=0, target=4)

print(f"Shortest path between node 0 and 4: {path}")

# Draw the graph
nx.draw(G, with_labels=True)
plt.show()
