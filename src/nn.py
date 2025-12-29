import os
import sys
import random

from demo import render_computation_graph

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../core/build/"))
)

from minigrad import Value, Neuron


x = [Value(2.0), Value(3.0)]
n = Neuron(len(x))
out = n(x)
print(out)
graph = out.exportGraph()
nodes = graph["nodes"]
edges = graph["edges"]

render_computation_graph(nodes, edges, "computation_graph_NN")
