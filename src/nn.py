import os
import sys
import random

from demo import render_computation_graph

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
Xv: list[list[Value]] = []
for row in X:  # X is (n_samples, n_features)
    Xv.append([Value(float(xi)) for xi in row])
yv: list[Value] = [Value(v) for v in y]

X_train, X_test, y_train, y_test = train_test_split(Xv, yv, test_size=0.25)
# print(len(X_train), len(X_test), len(y_train), len(y_test))

lr, epoch = 0.01, 1000
n_in = X_train[0].__len__()
n_out = 3
n1, n2, n3 = Neuron(n_in), Neuron(n_in), Neuron(n_in)  # input size = 4
layer = Layer([n1, n2, n3])
mlp = MLP([layer])

for _ in range(epoch):
    total_loss = Value(0.0)
    for xi, yi in zip(X_train, y_train):
        # forward pass
        y_pred_list = mlp(xi)
        y_target = [Value(yi.getData()) for _ in range(n_out)]
        # y_pred = y_pred_list[0]
        # loss
        # loss = (yi - y_pred) * (yi - y_pred)
        loss = sum(
            (
                (y_target[i] - y_pred_list[i]) * (y_target[i] - y_pred_list[i])
                for i in range(n_out)
            ),
            start=Value(0.0),
        )

        total_loss = total_loss + loss

    # zero_grad
    mlp.zero_grad()
    # bacward
    total_loss.backPropogate()
    for p in mlp.parameters():
        p.setData(p.getData() - lr * p.getGrad())

    if _ % 100 == 0:
        print(f"epoch {_} loss =", total_loss.getData())


# Predictions
preds = []
for xi in X_test:
    y_pred = mlp(xi)  # list of outputs (1 neuron in your case)
    pred_class = 0 if y_pred[0].getData() < 0.5 else 1
    preds.append(pred_class)

# Actual values
actual = [int(yt.getData()) for yt in y_test]

from sklearn.metrics import accuracy_score

acc = accuracy_score(actual, preds)
print(f"Accuracy on test set: {acc*100:.2f}%")
print("\n\n", actual, "\n", preds)
