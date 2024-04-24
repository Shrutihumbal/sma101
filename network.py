import networkx as nx
import csv
import matplotlib.pyplot as plt


def load_facebook_network(file_path):
    with open(file_path, "r") as file:
        edges = [line.strip().split() for line in file.readlines()]
    return edges


def compute_centrality_measures(graph):
    degree_centrality = nx.degree_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    edge_betweenness_centrality = nx.edge_betweenness_centrality(graph)
    eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000)
    return (
        degree_centrality,
        betweenness_centrality,
        edge_betweenness_centrality,
        eigenvector_centrality,
    )


def find_bridges(graph):
    bridges = list(nx.bridges(graph))
    return bridges


def main():
    facebook_network = load_facebook_network("facebook_networkdata.txt")
    G = nx.Graph()
    G.add_edges_from(facebook_network)
    nx.draw(G)
    plt.show()
    
    (
        degree_centrality,
        betweenness_centrality,
        edge_betweenness_centrality,
        eigenvector_centrality,
    ) = compute_centrality_measures(G)
    print("Degree Centrality:")
    for node, centrality in degree_centrality.items():
        print(f"Node {node}: {centrality}")

    print()
    print("Betweenness Centrality:")
    for node, centrality in betweenness_centrality.items():
        print(f"Node {node}: {centrality}")

    print()
    print("Edge Betweenness Centrality:")
    for edge, centrality in edge_betweenness_centrality.items():
        print(f"Edge {edge}: {centrality}")

    print()
    print("Eigenvector Centrality:")
    for node, centrality in eigenvector_centrality.items():
        print(f"Node {node}: {centrality}")

    print()
    print("Bridges:")
    bridges = find_bridges(G)
    print(bridges)

if __name__ == "__main__":
    main()
