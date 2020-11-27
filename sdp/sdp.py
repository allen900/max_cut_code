import time, itertools, random
import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from utils.cut import Cut
from utils.graph import Graph



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
    # print(len(graph.nodes()))
    adjacency = nx.linalg.adjacency_matrix(graph)
    adjacency = adjacency.toarray()

    start_time = time.time()
    solution = _solve_cut_vector_program(adjacency)
    sides = _recover_cut(solution)

    nodes = list(graph.nodes)
    left = {vertex for side, vertex in zip(sides, nodes) if side < 0}
    right = {vertex for side, vertex in zip(sides, nodes) if side >= 0}
    duration = time.time() - start_time

    return Cut(left, right), duration

def _solve_cut_vector_program(adjacency):
    """
    :param adjacency: (np.ndarray) A square matrix representing the adjacency
        matrix of an undirected graph with no self-loops. Therefore, the matrix
        must be symmetric with zeros along its diagonal.
    :return: (np.ndarray) A matrix whose columns represents the vectors assigned
        to each vertex to maximize the MAX-CUT semi-definite program (SDP)
        objective.
    """
    print("Creating adjacency matrix...")
    size = len(adjacency)
    ones_matrix = np.ones((size, size))
    products = cp.Variable((size, size), PSD=True)
    cut_size = 0.5 * cp.sum(cp.multiply(adjacency, ones_matrix - products))
    # print(cut_size)
    print("Initializing cvx problem...")
    objective = cp.Maximize(cut_size)
    constraints = [cp.diag(products) == 1]
    problem = cp.Problem(objective, constraints)
    for var in problem.variables():
        print(var.shape[0])
    print("Solving problem...")
    problem.solve()
    print("Done!")

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
    # graph, cut = sbm_graph(GRAPH_SIZE, WITHIN, BETWEEN)
    # visualize_cut(graph, cut)

    # print('Planted Cut')
    # print('Expected Size:', GRAPH_SIZE * GRAPH_SIZE * BETWEEN / 4)
    # print('Real Size:', cut.evaluate_cut_size(graph))

    # sdp_cut = goemans_williamson_weighted(graph)
    # visualize_cut(graph, sdp_cut)

    graph = Graph("data/musae_git_edges.csv")
    sdp_cut, duration = goemans_williamson_weighted(graph.graph)

    print('Goemans-Williamson Performance')
    print('Cut size:', sdp_cut.evaluate_cut_size(graph.graph))
    print('Running time:', duration)
