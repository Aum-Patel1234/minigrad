#include "value.h"
#include "consts.h"
#include <cmath>
#include <iostream>

Value::Value(double data)
    : data(data), prevNodes(), op(), label(), grad(0.0), backward(nullptr) {}
Value::Value(double data, const std::vector<const Value *> &prev)
    : data(data), prevNodes(prev), op(), label(), grad(0.0), backward(nullptr) {
}
Value::Value(double data, const std::vector<const Value *> &prev,
             const std::string_view op)
    : data(data), prevNodes(prev), op(op), label(), grad(0.0),
      backward(nullptr) {}
Value::Value(double data, const std::vector<const Value *> &prev,
             const std::string_view op, const std::string label)
    : data(data), prevNodes(prev), op(op), label(label), grad(0.0),
      backward(nullptr) {}

double Value::getData() const { return this->data; }
double Value::getGrad() const { return this->grad; }
std::string_view Value::getLabel() const { return this->label; }
std::string_view Value::getOp() const { return this->op; }

Value Value::operator+(const Value &other) const {
  Value out(this->data + other.data, {this, &other}, OP_ADD);

  out.backward = [](const Value *self) {
    const Value *a = self->prevNodes[0];
    const Value *b = self->prevNodes[1];
    a->grad += 1.0 * self->grad;
    b->grad += 1.0 * self->grad;
  };

  return out;
}

Value Value::operator-(const Value &other) const {
  Value out(this->data - other.data, {this, &other}, OP_SUB);

  out.backward = [](const Value *self) {
    const Value *a = self->prevNodes[0];
    const Value *b = self->prevNodes[1];
    a->grad += 1.0 * self->grad;
    b->grad += -1.0 * self->grad;
  };
  return out;
}

Value Value::operator*(const Value &other) const {
  Value out(this->data * other.data, {this, &other}, OP_MUL);

  out.backward = [](const Value *self) {
    const Value *a = self->prevNodes[0];
    const Value *b = self->prevNodes[1];
    a->grad += b->data * self->grad;
    b->grad += a->data * self->grad;
  };
  return out;
}

Value Value::operator/(const Value &other) const {
  Value out(this->data / other.data, {this, &other}, OP_DIV);
  out.backward = [](const Value *self) {
    const Value *a = self->prevNodes[0];
    const Value *b = self->prevNodes[1];
    // d(a/b)/da = 1/b
    a->grad += (1.0 / b->data) * self->grad;
    // d(a/b)/db = -a / (b^2)
    b->grad += (-a->data / (b->data * b->data)) * self->grad;
  };
  return out;
}

Value Value::tanh() const {
  Value out = Value(std::tanh(this->data), {this}, OP_TANH);

  out.backward = [](const Value *self) {
    const Value *a = self->prevNodes[0];
    // derivative of tanh is 1 - out^2
    a->grad += (1.0 - (self->data * self->data)) * self->grad;
  };
  return out;
}

Value Value::pow(const int n) const {
  Value out = Value(std::pow(this->data, n), {this}, OP_POW);

  out.backward = [n](const Value *self) {
    const Value *a = self->prevNodes[0];
    a->grad = n * std::pow(a->data, n - 1) * self->grad;
  };

  return out;
}

std::ostream &operator<<(std::ostream &os, const Value &val) {
  os << "Value(data=" << val.data << ")\n";
  return os;
}

void Value::buildGraph(
    std::vector<const Value *> &nodes,
    std::vector<std::pair<const Value *, const Value *>> &edges) const {
  std::queue<const Value *> q;
  std::unordered_set<const Value *> visited;
  q.push(this);
  visited.insert(this);

  while (!q.empty()) {
    auto top = q.front();
    nodes.push_back(top);
    q.pop();

    for (auto &node : top->prevNodes) {
      edges.push_back({node, top});
      if (visited.insert(node).second)
        q.push(node);
    }
  }
}

void Value::backPropogate() {
  this->grad = 1.0;
  std::vector<const Value *> topo;
  std::unordered_set<const Value *> visited;

  std::function<void(const Value *)> buildTopo = [&](const Value *node) {
    if (visited.find(node) != visited.end())
      return;
    visited.insert(node);
    for (auto prev : node->prevNodes)
      buildTopo(prev);
    topo.push_back(node);
  };

  buildTopo(this);

  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    if ((*it)->backward)
      (*it)->backward(*it);
  }
}
