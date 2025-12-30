#pragma once

#include "value.h"
#include <cstddef>
#include <memory>
#include <vector>

class Module {
public:
  virtual void zero_grad() {
    for (auto &p : parameters())
      p->zeroGrad();
  }
  virtual std::vector<std::shared_ptr<Value>> parameters() const = 0;
};

class Neuron : public Module {
private:
  std::vector<std::shared_ptr<Value>> w;
  std::shared_ptr<Value> b;

public:
  Neuron(size_t inputSize);
  std::shared_ptr<Value>
  operator()(const std::vector<std::shared_ptr<Value>> &input) const;

  std::vector<std::shared_ptr<Value>> parameters() const override;
};

class Layer : public Module {
private:
  std::vector<Neuron> neurons;

public:
  Layer(const std::vector<Neuron> &neurons);

  std::vector<std::shared_ptr<Value>>
  operator()(const std::vector<std::shared_ptr<Value>> &input) const;

  std::vector<std::shared_ptr<Value>> parameters() const override;
};

class MLP : public Module {
private:
  std::vector<Layer> layers;

public:
  MLP(const std::vector<Layer> &layers);
  std::vector<std::shared_ptr<Value>>
  operator()(const std::vector<std::shared_ptr<Value>> &input) const;

  std::vector<std::shared_ptr<Value>> parameters() const override;
};
