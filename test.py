import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
import Graph_Sampling

# df = pd.read_csv('data/as20000102.txt', sep='\t', header=None, skiprows=4,encoding='utf-8')
# df = pd.read_csv('data/musae_git_edges.csv', header=None, skiprows=1, encoding='utf-8')
# df = pd.read_csv('data/roadNet-CA.txt', sep='\t', header=None,
#                  skiprows=4, encoding='utf-8')
df = pd.read_csv('data/EIA930_INTERCHANGE_2020_Jan_Jun.csv', encoding='utf-8')
df = df.loc[df['Data Date']=='01/10/2020']
df = df.filter(items=['Balancing Authority', 'Directly Interconnected Balancing Authority'])
# df.columns = ['a', 'b']
G = nx.from_pandas_edgelist(df, 'Balancing Authority', 'Directly Interconnected Balancing Authority')
G.remove_edges_from(nx.selfloop_edges(G))
print(nx.transitivity(G))

# k = 2000
# sampled_nodes = random.sample(G.nodes, k)
# sampled_graph = G.subgraph(sampled_nodes)
# print(len(sampled_graph.nodes()))
# print(len(sampled_graph.edges()))

obj = Graph_Sampling.SRW_RWF_ISRW()
sampled_graph = obj.random_walk_induced_graph_sampling(G, 25)
print(nx.transitivity(sampled_graph))
print(len(sampled_graph.nodes()))
print(len(sampled_graph.edges()))

# print(sampled_graph.edges())
# print(sorted(sampled_graph))
# x = sorted(sampled_graph)
# mapping = {}
# for i in range(len(x)):
#     mapping[x[i]] = i
# # print(mapping)
# H = nx.relabel_nodes(sampled_graph, mapping)
# print(sorted(H))
