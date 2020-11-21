import time, itertools, random
import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
from utils.cut import Cut

class Show:
    def __init__(self, size, within, between, c_left, c_right):
        self.within = within
        self.between = between
        self.size = size
        self.graph, self.cut = self.sbm_graph()
        self.c_left = c_left
        self.c_right = c_right

    def stochastic_block_on_cut(self, cut):
        """
        Returns a graph drawn from the Stochastic Block Model, on the vertices
        in CUT. Every edge between pairs of vertices in CUT.LEFT and CUT.RIGHT is
        present independently with probability WITHIN; edges between sides are
        similarly present independently with probability BETWEEN.
        :param cut: (structures.cut.Cut) A cut which represents the vertices in
            each of the two communities. Traditionally, the size of each side is
            exactly half the total number of vertices in the graph, denoted n.
        :param within: (float) The probability an edge exists between two vertices
            in the same community, denoted p. Must be between 0 and 1 inclusive.
        :param between: (float) The probability of each edge between two vertices
            in different communities, denoted q. Must be between 0 and 1 inclusive.
        :return: (nx.classes.graph.Graph) A graph drawn according to the Stochastic
            Block Model over the cut.
        """
        graph = nx.Graph()
        graph.add_nodes_from(cut.vertices)

        for side in (cut.left, cut.right):
            for start, end in itertools.combinations(side, 2):
                if random.random() < self.within:
                    graph.add_edge(start, end)

        for start in cut.left:
            for end in cut.right:
                if random.random() < self.between:
                    graph.add_edge(start, end)

        return graph

    def sbm_graph(self):
        half = int(self.size / 2)
        left_side = np.random.choice(self.size, half, replace=False)
        left_side = set(left_side)

        cut = Cut(left_side, set())
        for vertex in range(self.size):
            if vertex not in left_side:
                cut.right.add(vertex)

        graph = self.stochastic_block_on_cut(cut)
        return graph, cut

    def visualize_cut(self, cut):
        colors = []
        for vertex in self.graph.nodes:
            color = self.c_left if vertex in cut.left else self.c_right
            colors.append(color)

        nx.draw(self.graph, node_color=colors)
        plt.show()