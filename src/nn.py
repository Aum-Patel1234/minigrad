import os
import sys
import random

from demo import render_computation_graph
from sklearn.metrics import accuracy_score

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../core/build/"))
)

from minigrad import Value, Neuron, Layer, MLP


x = [Value(2.0), Value(3.0)]
n = Neuron(len(x))
out = n(x)
print(out)
graph = out.exportGraph()
nodes = graph["nodes"]
edges = graph["edges"]

render_computation_graph(nodes, edges, "computation_graph_NN")

# test real data
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# convert to Value type
Xv = [[Value(float(xi)) for xi in row] for row in X]
yv: list[Value] = [Value(v) for v in y]
assert len(Xv) == len(X)
assert len(yv) == len(y)

X_train, X_test, y_train, y_test = train_test_split(Xv, yv, test_size=0.25)
# print(len(X_train), len(X_test), len(y_train), len(y_test))

# Architecture: 2 → 16 → 1
n_in = X_train[0].__len__()
hidden = Layer([Neuron(n_in) for _ in range(16)])
out = Layer([Neuron(16)])
mlp = MLP([hidden, out])
lr, epoch = 0.01, 500

for _ in range(epoch):
    total_loss = Value(0.0)

    for xi, yi in zip(X_train, y_train):
        # Forward
        y_pred = mlp(xi)[0]
        y_target = Value(1.0 if yi.getData() == 1 else -1.0)
        # loss
        loss = (y_pred - y_target) * (y_pred - y_target)
        total_loss = total_loss + loss

    # zero_grad
    mlp.zero_grad()
    # bacward
    total_loss.backPropogate()

    for p in mlp.parameters():
        p.setData(p.getData() - lr * p.getGrad())

    if _ % 100 == 0:
        print(f"epoch {_} loss =", total_loss.getData())

print(mlp.parameters()[0].getGrad())

# Predictions
preds = []
raw_preds = []

for xi in X_test:
    y_pred = mlp(xi)[0]
    raw_preds.append(y_pred.getData())

    # tanh output → decision boundary at 0
    pred_class = 1 if y_pred.getData() > 0 else 0
    preds.append(pred_class)

actual = [int(y.getData()) for y in y_test]

acc = accuracy_score(actual, preds)
print(f"\nAccuracy on test set: {acc * 100:.2f}%")

print("\nActual:", actual)
print("Preds: ", preds[:10])
print("Raw:   ", raw_preds[:10])


# ==========================
# SMALL GRAPH (2 SAMPLES)
# ==========================

VISIBLE_SAMPLES = 2

mlp.zero_grad()
small_loss = Value(0.0)

for xi, yi in list(zip(X_train, y_train))[:VISIBLE_SAMPLES]:
    y_pred = mlp(xi)[0]
    y_target = Value(1.0 if yi.getData() == 1 else -1.0)

    loss = (y_pred - y_target) * (y_pred - y_target)
    small_loss = small_loss + loss

# backward
small_loss.backPropogate()

# render clean graph
graph = small_loss.exportGraph()
render_computation_graph(graph["nodes"], graph["edges"], "mlp_loss_graph_2_samples")
