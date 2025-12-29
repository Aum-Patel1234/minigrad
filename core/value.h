#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <ostream>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

class Value : public std::enable_shared_from_this<Value> {
private:
  double data;
  std::vector<std::weak_ptr<Value>> prevNodes;
  std::string op, label;
  mutable double grad;
  std::function<void()> backward;

public:
  Value(double data);
  Value(double data, const std::vector<std::weak_ptr<Value>> &prevNodes);
  Value(double data, const std::vector<std::weak_ptr<Value>> &prevNodes,
        const std::string_view op);
  Value(double data, const std::vector<std::weak_ptr<Value>> &prevNodes,
        const std::string_view op, const std::string &label);
  ~Value() {
#ifndef NDEBUG
    std::cerr << "[Value destroyed] data=" << data << "\n";
#endif
  }

  void setData(double val);
  double getData() const;
  double getGrad() const;
  std::string_view getLabel() const;
  std::string_view getOp() const;
  long getRefCount() const;

  void buildGraph(
      std::vector<std::shared_ptr<Value>> &nodes,
      std::vector<std::pair<std::shared_ptr<Value>, std::shared_ptr<Value>>>
          &edges);

  std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &other);
  std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &other);
  std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &other);
  std::shared_ptr<Value> operator/(const std::shared_ptr<Value> &other);
  std::shared_ptr<Value> tanh();
  std::shared_ptr<Value> relu();
  std::shared_ptr<Value> pow(int n);

  friend std::ostream &operator<<(std::ostream &os,
                                  const std::shared_ptr<Value> &val);

  void backPropogate();
};
