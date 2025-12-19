#include "value.h"
#include "consts.h"
#include <queue>
#include <string_view>

Value::Value(double data) : data(data), grad(0) {}
Value::Value(double data, const std::unordered_set<Value *> &prevNodes)
    : data(data), prevNodes(prevNodes) {}
Value::Value(double data, const std::unordered_set<Value *> &prevNodes,
             const std::string_view op)
    : data(data), op(op), prevNodes(prevNodes) {}
Value::Value(double data, const std::unordered_set<Value *> &prevNodes,
             const std::string_view op, const std::string label)
    : data(data), prevNodes(prevNodes), op(op), label(label) {}

double Value::getData() const { return this->data; }

std::string_view Value::getLabel() const { return this->label; }
std::string_view Value::getOp() const { return this->op; }

Value Value::operator+(const Value &other) const {
  return Value(this->data + other.data,
               {const_cast<Value *>(this), const_cast<Value *>(&other)},
               OP_ADD);
}

Value Value::operator-(const Value &other) const {
  return Value(this->data - other.data,
               {const_cast<Value *>(this), const_cast<Value *>(&other)},
               OP_SUB);
}

Value Value::operator*(const Value &other) const {
  return Value(this->data * other.data,
               {const_cast<Value *>(this), const_cast<Value *>(&other)},
               OP_MUL);
}

Value Value::operator/(const Value &other) const {
  return Value(this->data / other.data,
               {const_cast<Value *>(this), const_cast<Value *>(&other)},
               OP_DIV);
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
