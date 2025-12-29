#include "value.h"
#include "consts.h"
#include <memory>
#include <vector>

Value::Value(double data)
    : data(data), prevNodes(), op(), label(), grad(0.0), backward(nullptr) {}
Value::Value(double data, const std::vector<std::weak_ptr<Value>> &prev)
    : data(data), prevNodes(prev), op(), label(), grad(0.0), backward(nullptr) {
}
Value::Value(double data, const std::vector<std::weak_ptr<Value>> &prev,
             const std::string_view op)
    : data(data), prevNodes(prev), op(op), label(), grad(0.0),
      backward(nullptr) {}
Value::Value(double data, const std::vector<std::weak_ptr<Value>> &prev,
             const std::string_view op, const std::string &label)
    : data(data), prevNodes(prev), op(op), label(label), grad(0.0),
      backward(nullptr) {}

void Value::setData(double val) { this->data = val; }

double Value::getData() const { return this->data; }
double Value::getGrad() const { return this->grad; }
std::string_view Value::getLabel() const { return this->label; }
std::string_view Value::getOp() const { return this->op; }
long Value::getRefCount() const {
  try {
    return shared_from_this().use_count();
  } catch (const std::bad_weak_ptr &) {
    return 0;
  }
}

std::shared_ptr<Value> Value::operator+(const std::shared_ptr<Value> &other) {
  auto self = shared_from_this();
  std::vector<std::weak_ptr<Value>> prev;
  prev.push_back(self);
  prev.push_back(other);
  std::shared_ptr<Value> out =
      std::make_shared<Value>(this->data + other->data, prev, OP_ADD);

  out->backward = [self, other, out]() {
    self->grad += 1.0 * out->grad;
    other->grad += 1.0 * out->grad;
  };

  return out;
}

std::shared_ptr<Value> Value::operator-(const std::shared_ptr<Value> &other) {
  auto self = shared_from_this();
  std::vector<std::weak_ptr<Value>> prev{self, other};
  auto out = std::make_shared<Value>(this->data - other->data, prev, OP_SUB);

  out->backward = [self, other, out]() {
    self->grad += 1.0 * out->grad;
    other->grad += -1.0 * out->grad;
  };
  return out;
}

std::shared_ptr<Value> Value::operator*(const std::shared_ptr<Value> &other) {
  auto self = shared_from_this();
  std::vector<std::weak_ptr<Value>> prev{self, other};
  auto out = std::make_shared<Value>(this->data * other->data, prev, OP_MUL);

  out->backward = [self, other, out]() {
    self->grad += other->data * out->grad;
    other->grad += self->data * out->grad;
  };
  return out;
}

std::shared_ptr<Value> Value::operator/(const std::shared_ptr<Value> &other) {
  auto self = shared_from_this();
  std::vector<std::weak_ptr<Value>> prev{self, other};
  auto out = std::make_shared<Value>(this->data / other->data, prev, OP_DIV);

  out->backward = [self, other, out]() {
    // d(a/b)/da = 1/b
    self->grad += (1.0 / other->data) * out->grad;
    // d(a/b)/db = -a / (b^2)
    other->grad += (-self->data / (other->data * other->data)) * out->grad;
  };
  return out;
}

std::shared_ptr<Value> Value::tanh() {
  auto self = shared_from_this();
  std::vector<std::weak_ptr<Value>> prev{self};
  auto out = std::make_shared<Value>(std::tanh(this->data), prev, OP_TANH);

  // TODO: convert args of this callback to weak_ptr as they do not need to
  // increase ref count  of the Value Obj
  out->backward = [self, out]() {
    // derivative of tanh is 1 - out^2
    self->grad += (1.0 - (out->data * out->data)) * out->grad;
  };
  return out;
}

std::shared_ptr<Value> Value::relu() {
  auto self = shared_from_this();
  std::vector<std::weak_ptr<Value>> prev{self};
  auto out = std::make_shared<Value>(std::max(0.0, this->data), prev, OP_RELU);

  out->backward = [self, out]() {
    self->grad += ((self->data < 0.0) ? 0 : 1) * out->grad;
  };

  return out;
}

std::shared_ptr<Value> Value::pow(int n) {
  auto self = shared_from_this();

  auto out = std::make_shared<Value>(
      std::pow(this->data, n), std::vector<std::weak_ptr<Value>>{self}, OP_POW);

  out->backward = [self, out, n]() {
    self->grad += n * std::pow(self->data, n - 1) * out->grad;
  };

  return out;
}

std::ostream &operator<<(std::ostream &os, const std::shared_ptr<Value> &val) {
  os << "Value(data=" << val->getData() << ", grad=" << val->getGrad() << ")";
  return os;
}

void Value::buildGraph(
    std::vector<std::shared_ptr<Value>> &nodes,
    std::vector<std::pair<std::shared_ptr<Value>, std::shared_ptr<Value>>>
        &edges) {
  std::queue<std::shared_ptr<Value>> q;
  std::unordered_set<Value *> visited;
  std::shared_ptr<Value> self = shared_from_this();
  q.push(self);
  visited.insert(self.get());

  while (!q.empty()) {
    auto top = q.front();
    nodes.push_back(top);
    q.pop();

    for (auto &wp : top->prevNodes) {
      if (auto parent = wp.lock()) {
        edges.push_back({parent, top});
        if (visited.insert(parent.get()).second)
          q.push(parent);
      }
    }
  }
}

void Value::backPropogate() {
  this->grad = 1.0;
  std::vector<std::shared_ptr<Value>> topo;
  std::unordered_set<Value *> visited;

  std::function<void(const std::shared_ptr<Value>)> buildTopo =
      [&](std::shared_ptr<Value> node) {
        if (visited.find(node.get()) != visited.end())
          return;
        visited.insert(node.get());
        for (auto prev : node->prevNodes) {
          if (prev.lock())
            buildTopo(prev.lock());
        }
        topo.push_back(node);
      };

  // NOTE: as it is in reverse i.e starting from front to back no need to
  // reverse as it automatically acts like a stack
  buildTopo(shared_from_this());

  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    if ((*it)->backward)
      (*it)->backward();
  }
}
