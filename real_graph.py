import time
import itertools
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
# GRAPH_SIZES = [10, 50, 100, 200]
# GRAPH_SIZES = [5, 10, 20, 30]


def random_cut_alg(graph): return randomized.random_cut(graph, 0.5)


greedy_cut_alg = greedy.greedy_max_cut
sdp_cut_alg = sdp.sdp_cut
quantum = qaoa_2.qaoa

# for classic


def get_graph_git():
    df = pd.read_csv('data/musae_git_edges.csv', encoding='utf-8')

    # targeted random sampling
    x = np.random.randint(1, 37471, size=25)
    filtered = df[df.id_1.isin(x) & df.id_2.isin(x)]
    print(filtered)

    graph = nx.convert_matrix.from_pandas_edgelist(filtered, 'id_1', 'id_2')

    return graph

# for quantum and classic


def get_graph_road():
    df = pd.read_csv('data/roadNet-CA.txt', encoding='utf-8')

    # targeted random sampling
    x = np.random.randint(1, 37471, size=20)
    filtered = df[df.id_1.isin(x) & df.id_2.isin(x)]
    print(filtered)

    graph = nx.convert_matrix.from_pandas_edgelist(filtered, 'id_1', 'id_2')

    return graph

# qaoa & classic


def get_graph_eia():
    df = pd.read_csv(
        'data/EIA930_INTERCHANGE_2020_Jan_Jun.csv', encoding='utf-8')

    # targeted random sampling
    x = np.random.randint(1, 37471, size=20)
    filtered = df[df.id_1.isin(x) & df.id_2.isin(x)]
    print(filtered)

    graph = nx.convert_matrix.from_pandas_edgelist(filtered, 'id_1', 'id_2')

    return graph

# classic


def get_graph_as():
    df = pd.read_csv('data/as20000102.txt', sep='\t',
                     header=None, skiprows=4, encoding='utf-8')
    df.columns = ['a', 'b']
    # targeted random sampling
    # x = np.random.randint(1, 37471, size=20)
    # filtered = df[df.id_1.isin(x) & df.id_2.isin(x)]
    # print(filtered)

    graph = nx.convert_matrix.from_pandas_edgelist(df, 'a', 'b')

    return graph


def average_performance(algorithm, trials=1):
    times, outputs = [], []
    for _ in range(trials):
        # graph = graph_generator()
        graph = get_graph_as()

        # start = time.clock()
        result, d = algorithm(graph)
        # end = time.clock()
        # elapsed = end - start

        times.append(d)
        if algorithm == quantum:
            outputs.append(result)
        else:
            outputs.append(result.evaluate_cut_size(graph))

    return {
        'trials': trials,
        'time': np.mean(times),
        'output': np.mean(outputs)
    }


upper_bounds = []
random_results = []
greedy_results = []
sdp_results = []
qaoa_results = []
GRAPH_SIZE = 6474
# for size in GRAPH_SIZES:
random_results.append(average_performance(random_cut_alg))
greedy_results.append(average_performance(greedy_cut_alg))
# sdp_results.append(average_performance(sdp_cut_alg))
# qaoa_results.append(average_performance(quantum))

# PLOTTING_OPTIONS = {
#     'title': 'Cut Size vs Graph Size',
#     'legend': [
#         'Random Cut',
#         # 'Greedy Cut',
#         # 'SDP Cut',
#         # 'QAOA',
#     ]
# }

# plt.plot(GRAPH_SIZES, [result['output'] for result in random_results])
# # plt.plot(GRAPH_SIZES, [result['output'] for result in greedy_results])
# # plt.plot(GRAPH_SIZES, [result['output'] for result in sdp_results])
# # plt.plot(GRAPH_SIZES, [result['output'] for result in qaoa_results])
# # plt.plot(GRAPH_SIZES, [result['output'] for result in upper_bounds])
# plt.title(PLOTTING_OPTIONS['title'])
# plt.legend(PLOTTING_OPTIONS['legend'])
# plt.show()


rows = []
# for pos in range(len(GRAPH_SIZES)):
rows.append([
    GRAPH_SIZE,
    # random_results[0]['output'],
    random_results[0]['output'],
    # sdp_results[pos]['output'],
    # qaoa_results[pos]['output'],
    # upper_bounds[pos]['output'],
    random_results[0]['time']
],
    [
    greedy_results[0]['output'],
    greedy_results[0]['time']
])

table = pd.DataFrame(rows)
# table.columns = ['Graph Size', 'Random', 'Greedy', 'SDP','Upper_bounds']
table.columns = ['cut', 'time']
table.rows = ['Random', 'Greedy']
print(table)
