#pragma once

#include <cmath>
#include <functional>
#include <ostream>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

class Value {
private:
  double data;
  std::vector<const Value *> prevNodes;
  std::string op, label;
  mutable double grad;
  std::function<void(const Value *)> backward;

public:
  Value(double data);
  Value(double data, const std::vector<const Value *> &prevNodes);
  Value(double data, const std::vector<const Value *> &prevNodes,
        const std::string_view op);
  Value(double data, const std::vector<const Value *> &prevNodes,
        const std::string_view op, const std::string label);

  double getData() const;
  double getGrad() const;
  std::string_view getLabel() const;
  std::string_view getOp() const;

  void
  buildGraph(std::vector<const Value *> &nodes,
             std::vector<std::pair<const Value *, const Value *>> &edges) const;

  Value operator+(const Value &other) const;
  Value operator-(const Value &other) const;
  Value operator*(const Value &other) const;
  Value operator/(const Value &other) const;
  Value tanh() const;
  Value relu() const;
  Value pow(const int n) const;

  // friend allows a function or class to access private and protected members
  friend std::ostream &operator<<(std::ostream &os, const Value &val);

  void backPropogate();
};
