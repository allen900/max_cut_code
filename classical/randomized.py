import time
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from utils.graph import Graph
from utils.cut import Cut
# print(sys.path)


def random_cut(graph, probability):

    size = len(graph)
    sides = np.random.binomial(1, probability, size)

    nodes = list(graph.nodes)

    # algorithm core
    start_time = time.time()
    left = {vertex for side, vertex in zip(sides, nodes) if side == 0}
    right = {vertex for side, vertex in zip(sides, nodes) if side == 1}
    duration = time.time()-start_time

    return Cut(left, right), duration


if __name__ == "__main__":
    # graph, cut = sbm_graph(GRAPH_SIZE, WITHIN, BETWEEN)
    # visualize_cut(graph, cut)

    graph = Graph("data/musae_git_edges.csv", is_real=False)
    random_cut, duration = random_cut(graph.graph, 0.5)
    # graph.visualize_cut(random_cut)
    # visualize_cut(graph.graph, random_cut)

    print('------Random Cut Cost------')
    print(graph.s, 'nodes')
    print("Expected size:", int(graph.s * graph.s * graph.b / 4))
    print('Cut size:', random_cut.evaluate_cut_size(graph.graph))
    print('Running time:', duration)
