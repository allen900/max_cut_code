import time
import itertools
import random
import numpy as np
import networkx as nx
import pandas as pd
import Graph_Sampling
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
    df = pd.read_csv('data/musae_git_edges.csv', header=None,
                     skiprows=1, encoding='utf-8')
    df.columns = ['a', 'b']
    size = 25
    G = nx.from_pandas_edgelist(df, 'a', 'b')
    G.remove_edges_from(nx.selfloop_edges(G))
    obj = Graph_Sampling.SRW_RWF_ISRW()
    E = 0
    while(E < size*2):
        sampled_graph = obj.random_walk_induced_graph_sampling(G, size)
        E = len(sampled_graph.edges())

    return relabel(sampled_graph), size

# for quantum and classic


def get_graph_road():
    df = pd.read_csv('data/roadNet-CA.txt', sep='\t',
                     header=None, skiprows=4, encoding='utf-8')
    df.columns = ['a', 'b']
    size = 25
    G = nx.from_pandas_edgelist(df, 'a', 'b')
    G.remove_edges_from(nx.selfloop_edges(G))
    obj = Graph_Sampling.SRW_RWF_ISRW()
    E = 0
    while(E < 30):
        print('processing new sample...')
        sampled_graph = obj.random_walk_induced_graph_sampling(G, size)
        E = len(sampled_graph.edges())

    return relabel(sampled_graph), size

# qaoa & classic


def get_graph_eia():
    df = pd.read_csv('data/EIA930_INTERCHANGE_2020_Jan_Jun.csv', encoding='utf-8')
    df = df.loc[df['Data Date']=='01/10/2020']
    df = df.filter(items=['Balancing Authority', 'Directly Interconnected Balancing Authority'])
    size = 25
    G = nx.from_pandas_edgelist(df, 'Balancing Authority', 'Directly Interconnected Balancing Authority')
    G.remove_edges_from(nx.selfloop_edges(G))
    obj = Graph_Sampling.SRW_RWF_ISRW()
    E = 0
    while(E < size*2):
        sampled_graph = obj.random_walk_induced_graph_sampling(G, size)
        E = len(sampled_graph.edges())

    return relabel(sampled_graph), size

# classic & q


def get_graph_as():
    df = pd.read_csv('data/as20000102.txt', sep='\t',
                     header=None, skiprows=4, encoding='utf-8')
    df.columns = ['a', 'b']
    size = 25
    G = nx.from_pandas_edgelist(df, 'a', 'b')
    G.remove_edges_from(nx.selfloop_edges(G))
    obj = Graph_Sampling.SRW_RWF_ISRW()
    E = 0
    while(E < size*2):
        print('processing new sample...')
        sampled_graph = obj.random_walk_induced_graph_sampling(G, size)
        E = len(sampled_graph.edges())

    return relabel(sampled_graph), size


def relabel(G):
    x = sorted(G)
    mapping = {}
    for i in range(len(x)):
        mapping[x[i]] = i
    H = nx.relabel_nodes(G, mapping)
    print(sorted(H))
    nx.draw(H, with_labels=True)
    plt.show()
    return H


def average_performance(algorithm, G, trials=1):
    times, outputs = [], []
    for _ in range(trials):
        # graph = graph_generator()
        # graph = get_graph_as()

        # start = time.clock()
        result, d = algorithm(G)
        # end = time.clock()
        # elapsed = end - start

        if algorithm == quantum:
            outputs.append(result)
        else:
            outputs.append(result.evaluate_cut_size(G))
        times.append(d)

    return {
        'time': np.mean(times),
        'output': np.mean(outputs)
    }


upper_bounds = []
random_results = []
greedy_results = []
sdp_results = []
qaoa_results = []


G, GRAPH_SIZE = get_graph_eia()
# for size in GRAPH_SIZES:
random_results.append(average_performance(random_cut_alg, G))
greedy_results.append(average_performance(greedy_cut_alg, G))
sdp_results.append(average_performance(sdp_cut_alg, G))
qaoa_results.append(average_performance(quantum, G))

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
    random_results[0]['output'],
    random_results[0]['time']
])

rows.append([
    greedy_results[0]['output'],
    greedy_results[0]['time']
])

rows.append([
    sdp_results[0]['output'],
    sdp_results[0]['time']
])

rows.append([
    qaoa_results[0]['output'],
    qaoa_results[0]['time']
])

table = pd.DataFrame(rows)
# table.columns = ['Graph Size', 'Random', 'Greedy', 'SDP','Upper_bounds']
table.columns = ['cut', 'time']
table.rows = ['Random', 'Greedy', 'SDP', 'QAOA']
print(table)
