import time, itertools, random
import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from utils.cut import Cut
from utils.show import Show



def goemans_williamson_weighted(graph):
    """
    Runs the Goemans-Williamson randomized 0.87856-approximation algorithm for
    MAX-CUT on the graph instance, returning the cut.
    :param graph: (nx.classes.graph.Graph) An undirected graph with no
        self-loops or multiple edges. The graph can either be weighted or
        unweighted, where each edge present is assigned an equal weight of 1.
    :return: (structures.cut.Cut) The cut returned by the algorithm as two
        sets, where each corresponds to a different side of the cut. Together,
        both sets contain all vertices in the graph, and each vertex is in
        exactly one of the two sets.
    """
    adjacency = nx.linalg.adjacency_matrix(graph)
    adjacency = adjacency.toarray()
    solution = _solve_cut_vector_program(adjacency)
    sides = _recover_cut(solution)

    nodes = list(graph.nodes)
    left = {vertex for side, vertex in zip(sides, nodes) if side < 0}
    right = {vertex for side, vertex in zip(sides, nodes) if side >= 0}
    return Cut(left, right)


def stochastic_block_on_cut(cut, within, between):
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
            if random.random() < within:
                graph.add_edge(start, end)

    for start in cut.left:
        for end in cut.right:
            if random.random() < between:
                graph.add_edge(start, end)

    return graph


def _solve_cut_vector_program(adjacency):
    """
    :param adjacency: (np.ndarray) A square matrix representing the adjacency
        matrix of an undirected graph with no self-loops. Therefore, the matrix
        must be symmetric with zeros along its diagonal.
    :return: (np.ndarray) A matrix whose columns represents the vectors assigned
        to each vertex to maximize the MAX-CUT semi-definite program (SDP)
        objective.
    """
    size = len(adjacency)
    ones_matrix = np.ones((size, size))
    products = cp.Variable((size, size), PSD=True)
    cut_size = 0.5 * cp.sum(cp.multiply(adjacency, ones_matrix - products))

    objective = cp.Maximize(cut_size)
    constraints = [cp.diag(products) == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    eigenvalues, eigenvectors = np.linalg.eigh(products.value)
    eigenvalues = np.maximum(eigenvalues, 0)
    diagonal_root = np.diag(np.sqrt(eigenvalues))
    assignment = diagonal_root @ eigenvectors.T
    return assignment


def _recover_cut(solution):
    """
    :param solution: (np.ndarray) A vector assignment of vertices, where each
        SOLUTION[:,i] corresponds to the vector associated with vertex i.
    :return: (np.ndarray) The cut from probabilistically rounding the
        solution, where -1 signifies left, +1 right, and 0 (which occurs almost
        surely never) either.
    """
    size = len(solution)
    partition = np.random.normal(size=size)
    projections = solution.T @ partition

    sides = np.sign(projections)
    return sides


def sbm_graph(size, within, between):
    half = int(size / 2)
    left_side = np.random.choice(size, half, replace=False)
    left_side = set(left_side)

    cut = Cut(left_side, set())
    for vertex in range(size):
        if vertex not in left_side:
            cut.right.add(vertex)

    graph = stochastic_block_on_cut(cut, within, between)
    return graph, cut


def visualize_cut(graph, cut):
    colors = []
    for vertex in graph.nodes:
        color = LEFT_COLOR if vertex in cut.left else RIGHT_COLOR
        colors.append(color)
    # %matplotlib inline
    nx.draw(graph, node_color=colors)
    plt.show()


def average_performance(graph_generator, algorithm, trials=50):
    times, outputs = [], []
    for _ in range(trials):
        graph = graph_generator()

        start = time.clock()
        result = algorithm(graph)
        end = time.clock()
        elapsed = end - start

        times.append(elapsed)
        outputs.append(result.evaluate_cut_size(graph))

    return {
        'trials': trials,
        'time': np.mean(times),
        'output': np.mean(outputs)
    }


if __name__ == "__main__":
    LEFT_COLOR = 'red'
    RIGHT_COLOR = 'skyblue'
    GRAPH_SIZE = 50
    WITHIN = 0.25
    BETWEEN = 0.75

    # graph, cut = sbm_graph(GRAPH_SIZE, WITHIN, BETWEEN)
    # visualize_cut(graph, cut)

    # print('Planted Cut')
    # print('Expected Size:', GRAPH_SIZE * GRAPH_SIZE * BETWEEN / 4)
    # print('Real Size:', cut.evaluate_cut_size(graph))

    # sdp_cut = goemans_williamson_weighted(graph)
    # visualize_cut(graph, sdp_cut)

    visualization = Show(GRAPH_SIZE, WITHIN, BETWEEN, LEFT_COLOR, RIGHT_COLOR)
    sdp_cut = goemans_williamson_weighted(visualization.graph)

    print('Goemans-Williamson Performance')
    print('Cut size:', sdp_cut.evaluate_cut_size(visualization.graph))
