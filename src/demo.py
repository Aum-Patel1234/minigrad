import os
import sys
from graphviz.graphs import Digraph

sys.path.append(os.path.abspath("../core/build/"))

import minigrad


from graphviz import Digraph


def render_computation_graph(
    nodes,
    edges,
    filename="computation_graph",
    format="png",
    rankdir="LR",
):
    """
    format: png | svg | ...
    rankdir: TB (top-bottom) | LR (left-right)
    """
    assert rankdir in ("LR", "TB")

    dot = Digraph(
        name="ComputationGraph",
        format=format,
        graph_attr={"rankdir": rankdir},
        node_attr={"shape": "record"},
    )

    # ---- nodes ----
    for v in nodes:
        vid = str(id(v))
        label = v.getLabel()
        grad = v.getGrad()

        # main value node with gradient
        node_label = (
            "{ %s | data %.4f | grad %.4f }" % (label, v.getData(), grad)
            if label
            else "{ data %.4f | grad %.4f }" % (v.getData(), grad)
        )

        dot.node(vid, node_label)

        # operation node (if exists)
        op = v.getOp()
        if op:
            op_id = vid + op
            dot.node(op_id, op, shape="circle")
            dot.edge(op_id, vid)

    # ---- edges ----
    for parent, child in edges:
        if child.getOp():
            dot.edge(str(id(parent)), str(id(child)) + child.getOp())
        else:
            dot.edge(str(id(parent)), str(id(child)))

    dot.render(filename, cleanup=True)
    print(f"Graph saved as {filename}.{format}")


def main():
    a = minigrad.Value(2)
    b = minigrad.Value(3)

    c = a + b
    d = a - b

    e = c * d

    f = e / a

    g = f.tanh()

    print("a =", a)
    print("b =", b)
    print("c = a + b =", c)
    print("d = a - b =", d)
    print("e = c * d =", e)
    print("f = e / a =", f)
    print("g = tanh(f) =", g)

    f.backPropogate()

    print("\nAfter backward:")
    print(f"a.grad = {a.getGrad()}")
    print(f"b.grad = {b.getGrad()}")
    print(f"c.grad = {c.getGrad()}")
    print(f"d.grad = {d.getGrad()}")
    print(f"e.grad = {e.getGrad()}")
    print(f"f.grad = {f.getGrad()}")
    print(f"g.grad = {g.getGrad()}")

    graph = f.exportGraph()
    nodes = graph["nodes"]
    edges = graph["edges"]

    render_computation_graph(nodes, edges)


if __name__ == "__main__":
    main()
