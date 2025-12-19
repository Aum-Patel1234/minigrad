#pragma once

#include <ostream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>
class Value {
private:
  double data;
  std::unordered_set<Value *> prevNodes;
  std::string op, label;
  double grad;

public:
  Value(double data);
  Value(double data, const std::unordered_set<Value *> &prevNodes);
  Value(double data, const std::unordered_set<Value *> &prevNodes,
        const std::string_view op);
  Value(double data, const std::unordered_set<Value *> &prevNodes,
        const std::string_view op, const std::string label);

  double getData() const;
  std::string_view getLabel() const;
  std::string_view getOp() const;

  void
  buildGraph(std::vector<const Value *> &nodes,
             std::vector<std::pair<const Value *, const Value *>> &edges) const;

  Value operator+(const Value &other) const;
  Value operator-(const Value &other) const;
  Value operator*(const Value &other) const;
  Value operator/(const Value &other) const;

  // friend allows a function or class to access private and protected members
  friend std::ostream &operator<<(std::ostream &os, const Value &val);
};
