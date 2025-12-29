import os
import gc
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
    n = minigrad.Value(22)
    c = a + b
    d = a - b

    e = c * d

    f = e / a

    g = f.tanh()
    h = f.pow(3)

    print("a =", a)
    print("b =", b)
    print("c = a + b =", c)
    print("d = a - b =", d)
    print("e = c * d =", e)
    print("f = e / a =", f)
    print("g = tanh(f) =", g)
    print("h = h ** 3 =", h)

    h.backPropogate()

    print("\nAfter backward:")
    print(f"a.grad = {a.getGrad()}")
    print(f"b.grad = {b.getGrad()}")
    print(f"c.grad = {c.getGrad()}")
    print(f"d.grad = {d.getGrad()}")
    print(f"e.grad = {e.getGrad()}")
    print(f"f.grad = {f.getGrad()}")

    graph = h.exportGraph()
    nodes = graph["nodes"]
    edges = graph["edges"]

    render_computation_graph(nodes, edges, "computation_graph_h")

    print("\n\n REF count of each shared ptr\n")
    print("Ref count (a) =", a.getRefCount())
    print("Ref count (b) =", b.getRefCount())
    print("Ref count (c) = a + b =", c.getRefCount())
    print("Ref Count d = a - b =", d.getRefCount())
    print("ref count e = c * d =", e.getRefCount())
    print("ref count f = e / a =", f.getRefCount())
    print("ref count g = tanh(f) =", g.getRefCount())
    print("ref count h = h ** 3 =", h.getRefCount())
    print("ref count n = ", n.getRefCount())

    del nodes
    del edges
    del graph

    del h
    del g
    del f
    del e
    del d
    del c
    del b
    del a

    del n
    # ref count n =  2 (ONe hold by python and one hold by Pybind11)
    # [Value destroyed] data=22
    # GC done

    gc.collect()
    print("GC done")


if __name__ == "__main__":
    main()
