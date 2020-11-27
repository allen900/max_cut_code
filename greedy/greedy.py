import time, itertools, random
import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
# print(sys.path)
from utils.cut import Cut
from utils.graph import Graph


def greedy_max_cut(graph):
    """
    Runs a greedy MAX-CUT approximation algorithm to partition the vertices of
    the graph into two sets. This greedy approach delivers an approximation
    ratio of 0.5.
    :param graph: (nx.classes.graph.Graph) An undirected graph with no
        self-loops or multiple edges. The graph can either be weighted or
        unweighted, where each edge present is assigned an equal weight of 1.
    :return: (structures.cut.Cut) The cut returned by the algorithm as two
        sets, where each corresponds to a different side of the cut. Together,
        both sets contain all vertices in the graph, and each vertex is in
        exactly one of the two sets.
    """
    cut = Cut(set(), set())
    start_time = time.time()
    for vertex in graph.nodes:
        l_neighbors = sum((adj in cut.left) for adj in graph.neighbors(vertex))
        r_neighbors = sum((adj in cut.right) for adj in graph.neighbors(vertex))
        if l_neighbors < r_neighbors:
            cut.left.add(vertex)
        else:
            cut.right.add(vertex)
    duration = time.time() - start_time
    
    return cut, duration

if __name__ == "__main__":
    graph = Graph("data/musae_git_edges.csv")
    greedy_max_cut, duration = greedy_max_cut(graph.graph)
    # graph.visualize_cut(random_cut)
    # visualize_cut(graph.graph, random_cut) 

    print('Greedy Cut Cost')
    print('Cut size:', greedy_max_cut.evaluate_cut_size(graph.graph))
    print('Running time:', duration)