import os
import sys
from graphviz.graphs import Digraph

sys.path.append(os.path.abspath("../core/build/"))

import minigrad


from graphviz import Digraph


def render_computation_graph(nodes, edges, filename="computation_graph"):
    dot = Digraph(comment="Computation Graph")

    # Add nodes
    for v in nodes:
        node_id = str(id(v))
        label = v.getLabel()
        if label:
            label_str = f"{label}\nValue({v.getData()})"
        else:
            label_str = f"Value({v.getData()})"
        dot.node(node_id, label_str)

    # Add edges with operation labels
    for parent, child in edges:
        op_label = child.getOp()
        dot.edge(str(id(parent)), str(id(child)), label=op_label)

    # Render to file
    dot.render(filename, format="png", cleanup=True)
    print(f"Graph saved as {filename}.png")


def main():
    a = minigrad.Value(2)
    b = minigrad.Value(3)

    c = a + b
    d = a - b

    e = c * d

    f = e / a

    print("a =", a)
    print("b =", b)
    print("c = a + b =", c)
    print("d = a - b =", d)
    print("e = c * d =", e)
    print("f = e / a =", f)

    graph = f.exportGraph()
    nodes = graph["nodes"]
    edges = graph["edges"]

    render_computation_graph(nodes, edges)


if __name__ == "__main__":
    main()
