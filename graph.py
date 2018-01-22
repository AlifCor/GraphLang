import numpy as np
import networkx as nx
import community
from matplotlib import pyplot as plt

def sparsity(m):
    return 1 - np.count_nonzero(m) / m.size

def np_to_nx(M, words_map=None):
    G = nx.from_numpy_matrix(M)
    if(words_map != None):
        words_map_inv = {e[1]:e[0] for e in words_map.items()}
        nx.set_node_attributes(G, "word", word_map_inversed)

    return G

def compute_betweenness(G, weight="weight"):
    betweenness = nx.betweenness_centrality(G, weight=weight)
    nx.set_node_attributes(G, "betweenness", betweenness)

    return betweenness

def scale_betweenness(betweenness, min_=10, max_=120):
    max_el = max(betweenness.items(), key=lambda el: el[1])
    mult = max_ / (max_el + min_)
    betweenness_scaled = {k: mult*v + min_ for k,v in betweenness.items()}

    return betweenness_scaled

def community_partition(G, weight="weight"):
    if(weight == "betweenness" and G.nodes()[0].get("betweenness") == None):
        compute_betweenness(G)

    return community.best_partition(Gn, weight=weight)

def communities(G, draw=True, cmap=None, pos=None, partition=None, betweenness_scaled=None):
    if(partition == None):
        partition = community_partition(G, weight="betweenness")
    if(betweenness_scaled == None):
        if(G.nodes()[0].get("betweenness") == None):
            betweenness = compute_betweenness(G, "betweenness")
        else:
            betweenness = nx.get_node_attributes(G, "betweenness")
        betweenness_scaled = scale_betweenness(betweenness)
    if(pos == None):
        pos = nx.spring_layout(G)

    if(draw and cmap):
        count = 0.
        for com in set(partition.values()):
            count = count + 1.
            list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
            sizes = [betweenness_scaled[node] for node in list_nodes]
            nx.draw_networkx_nodes(G, pos, list_nodes, node_size=sizes, node_color = cmap[com])

        nx.draw_networkx_edges(G, pos, alpha=0.05)

    return pos, partition, betweenness_scaled

def induced_graph(original_graph, induced_graph=None, draw=True, cmap=None, words_map_inv=None, pos=None, betweenness_scaled=None):
    if(induced_graph == None):
        induced_graph = community.induced_graph(partition, original_graph, weight=sizes)

    if(draw and cmap):
        if(pos == None):
            pos = nx.spring_layout(induced_graph)

        w = induced_graph.degree(weight="weight")

        sizes = [w[node] / 350 for node in induced_graph.nodes()]
        nx.draw(induced_graph, pos=pos, node_size=sizes, node_color=[cmap[n] for n in induced_graph.nodes()])

        labels = {}
        for com in induced_graph.nodes():
            rep = max([nodes for nodes in partition.keys() if partition[nodes] == com], key=lambda n: original_graph.degree(n, weight="weight"))
            labels[com] = words_map_inv[rep]

        nx.draw_networkx_labels(induced_graph, pos_gnc, labels, font_size=16)
    
    return induced_graph
