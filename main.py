import time, itertools
import random
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from classical import randomized, greedy, sdp
from qaoa import qaoa_2
from utils.graph import Graph
from utils.cut import Cut

# GRAPH_SIZES = [5, 10, 25, 50, 100]
GRAPH_SIZES = [10, 15, 20, 25]
# GRAPH_SIZES = [5, 10, 100, 200, 500]
# GRAPH_SIZE = 20
# WITHIN = 0.25
# BETWEEN = 0.75
# WITHIN = 0.5
# BETWEEN = 0.5

random_cut_alg = lambda graph: randomized.random_cut(graph, 0.5)
greedy_cut_alg = greedy.greedy_max_cut
sdp_cut_alg= sdp.sdp_cut
quantum = qaoa_2.qaoa

# def sbm_graph(size, within, between):
#     half = int(size / 2)
#     left_side = np.random.choice(size, half, replace=False)
#     left_side = set(left_side)
    
#     cut = Cut(left_side, set())
#     for vertex in range(size):
#         if vertex not in left_side:
#             cut.right.add(vertex)
    
#     graph = stochastic_block_on_cut(cut, within, between)
#     return graph, cut

# def stochastic_block_on_cut(cut, within, between):
#     graph = nx.Graph()
#     graph.add_nodes_from(cut.vertices)

#     for side in (cut.left, cut.right):
#         for start, end in itertools.combinations(side, 2):
#             if random.random() < within:
#                 graph.add_edge(start, end)

#     for start in cut.left:
#         for end in cut.right:
#             if random.random() < between:
#                 graph.add_edge(start, end)

#     return graph

def get_graph(size):
    graph = nx.gnp_random_graph(size, 0.5)
    # print(graph.nodes())
    return graph

def average_performance(size, algorithm, trials=50):
    times, outputs = [], []
    best = 0
    for _ in range(trials):
        graph = get_graph(size)
        if algorithm == quantum:
            best, result, d = algorithm(graph)
            outputs.append(result)
            times.append(d)
            break
        result,d = algorithm(graph)
        # print(size*size/4)
        times.append(d)
        outputs.append(result.evaluate_cut_size(graph))

    return {
        'trials': trials,
        'time': np.mean(times),
        'output': np.mean(outputs),
        'best':best
    }

upper_bounds = []
random_results = []
greedy_results = []
sdp_results = []
qaoa_results = []

for size in GRAPH_SIZES:
    print('processing size', size)
    # generator = lambda: sbm_graph(size, WITHIN, BETWEEN)[0]
    # generator = get_graph(size)
    random_results.append(average_performance(size, random_cut_alg))
    greedy_results.append(average_performance(size, greedy_cut_alg))
    sdp_results.append(average_performance(size, sdp_cut_alg))
    qaoa_results.append(average_performance(size, quantum))
    # upper_bounds.append(average_performance(size, sdp_cut_alg))

PLOTTING_OPTIONS = {
    'title': 'Cut Size vs Graph Size',
    'legend': [
        'Random Cut',
        'Greedy Cut',
        'SDP Cut',
        # 'QAOA avg',
        'QAOA',
        # 'Upper Bound'
    ]
}

plt.plot(GRAPH_SIZES, [result['output'] for result in random_results])
plt.plot(GRAPH_SIZES, [result['output'] for result in greedy_results])
plt.plot(GRAPH_SIZES, [result['output'] for result in sdp_results])
# plt.plot(GRAPH_SIZES, [result['output'] for result in qaoa_results])
plt.plot(GRAPH_SIZES, [result['best'] for result in qaoa_results])
# plt.plot(GRAPH_SIZES, [result['output'] for result in upper_bounds])
plt.title(PLOTTING_OPTIONS['title'],fontsize=20)
plt.legend(PLOTTING_OPTIONS['legend'],fontsize=20)
plt.show()


# PLOTTING_OPTIONS = {
#     'title': 'Running time vs Graph Size',
#     'legend': [
#         'SDP Cut',
#         'QAOA'
#     ]
# }

# plt.plot(GRAPH_SIZES, [result['time'] for result in random_results])
# plt.plot(GRAPH_SIZES, [result['time'] for result in greedy_results])
# plt.plot(GRAPH_SIZES, [result['time'] for result in sdp_results])
# # plt.plot(GRAPH_SIZES, [result['time'] for result in qaoa_results])
# plt.title(PLOTTING_OPTIONS['title'])
# plt.legend(PLOTTING_OPTIONS['legend'])
# plt.show()

# rows = []
# for pos in range(len(GRAPH_SIZES)):
#     rows.append([
#         GRAPH_SIZES[pos],
#         # random_results[pos]['output'],
#         # greedy_results[pos]['output'],
#         # sdp_results[pos]['output'],
#         qaoa_results[pos]['output'],
#         qaoa_results[pos]['best'],
#         upper_bounds[pos]['output'],
#         qaoa_results[pos]['best'] / upper_bounds[pos]['output'],
#         qaoa_results[pos]['time']
#     ])

# table = pd.DataFrame(rows)
# # table.columns = ['Graph Size', 'Random', 'Greedy', 'SDP', 'QAOA']
# table.columns = ['Graph Size', 'avg', 'best', 'Upper_bounds', 'ratio', 'time']
# # table.columns = ['Graph Size', 'Alg','time']
# print(table)
