import time, itertools, random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
# print(sys.path)
from utils.cut import Cut
from utils.show import Show


def random_cut(graph, probability):
    """
    :param graph: (nx.classes.graph.Graph) A NetworkX graph.
    :param probability: (float) A number in [0, 1] which gives the probability
        each vertex lies on the right side of the cut.
    :return: (structures.cut.Cut) The random cut which results from randomly
        assigning vertices to either side independently at random according
        to the probability given above.
    """

    size = len(graph)
    sides = np.random.binomial(1, probability, size)

    nodes = list(graph.nodes)
    left = {vertex for side, vertex in zip(sides, nodes) if side == 0}
    right = {vertex for side, vertex in zip(sides, nodes) if side == 1}
    # print(left)
    return Cut(left, right)

if __name__ == "__main__":
    LEFT_COLOR = 'red'
    RIGHT_COLOR = 'skyblue'
    GRAPH_SIZE = 10
    WITHIN = 0.25
    BETWEEN = 0.75


    # graph, cut = sbm_graph(GRAPH_SIZE, WITHIN, BETWEEN)
    # visualize_cut(graph, cut)

    visualization = Show(GRAPH_SIZE, WITHIN, BETWEEN, LEFT_COLOR, RIGHT_COLOR)
    random_cut = random_cut(visualization.graph, 0.5)
    # visualization.visualize_cut(random_cut)
    # visualize_cut(visualization.graph, random_cut)

    print('Random Cut Cost')
    print('Cut size:', random_cut.evaluate_cut_size(visualization.graph))
