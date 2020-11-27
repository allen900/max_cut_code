import time
import itertools
import random
import numpy as np
import cvxpy as cp
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from utils.cut import Cut


class Graph:
    def __init__(self, file, is_real):
        self.c_left = 'red'
        self.c_right = 'skyblue'
        self.file = file
        if is_real:
            self.graph = self.get_real_graph()
        else:
            self.graph = self.get_random_graph()

    def get_random_graph(self):
        
        

        return graph
    
    def get_real_graph(self):

        df = pd.read_csv(self.file, encoding="utf-8")
        df = df.head(500)
        # print(df.columns)
        graph = nx.convert_matrix.from_pandas_edgelist(df, "id_1", "id_2")

        return graph

    def visualize_cut(self, cut):
        colors = []
        for vertex in self.graph.nodes:
            color = self.c_left if vertex in cut.left else self.c_right
            colors.append(color)

        nx.draw(self.graph, node_color=colors)
        plt.show()
