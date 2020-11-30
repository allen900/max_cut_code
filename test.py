import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import collections
import random
import numpy as np

rng = np.random.default_rng()
x =  rng.choice(25000, size=100, replace=False)
df = pd.read_csv('data/as20000102.txt', sep='\t', header=None, skiprows=4,encoding='utf-8')
df.columns = ['a', 'b']
# print(x)
G = nx.from_pandas_edgelist(df, 'a', 'b')
# print(nx.transitivity(G))
print(df.shape[0])
# for _ in len(x):
#     filtered = df[df.id_1.isin(x) & df.id_2.isin(x)]
#     if filtered.shape[0] >= 10
#     print(filtered)

# S = [G.subgraph(c).copy() for c in nx.connected_components(G)]

# for subG in S:
#     print(len(subG.nodes()))

k = 2000
sampled_nodes = random.sample(G.nodes, k)
sampled_graph = G.subgraph(sampled_nodes)
print(len(sampled_graph.nodes()))
print(len(sampled_graph.edges()))