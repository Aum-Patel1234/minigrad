import os
import sys
import random

from demo import render_computation_graph

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../core/build/"))
)

import minigrad


class Neuron:
    def __init__(self, inputSize):
        self.w = [minigrad.Value(random.uniform(-1, 1)) for _ in range(inputSize)]
        self.b = minigrad.Value(random.uniform(-1, 1))

    def __call__(self, x: list[minigrad.Value]) -> minigrad.Value:
        out = self.b
        for wi, xi in zip(self.w, x):
            out = out + (wi * xi)
        return out.tanh()


x = [minigrad.Value(2.0), minigrad.Value(3.0)]
n = Neuron(len(x))
out = n(x)
print(out)
graph = out.exportGraph()
nodes = graph["nodes"]
edges = graph["edges"]

render_computation_graph(nodes, edges, "computation_graph_NN")
