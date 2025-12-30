#pragma once

#include "value.h"
#include <cstddef>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

class Module {
public:
  virtual ~Module() = default;
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
  // Make Neuron non-copyable to prevent accidental copies
  Neuron(const Neuron &) = delete;
  Neuron &operator=(const Neuron &) = delete;
  Neuron(size_t inputSize);
  std::shared_ptr<Value>
  operator()(const std::vector<std::shared_ptr<Value>> &input) const;

  std::vector<std::shared_ptr<Value>> parameters() const override;
};

class Layer : public Module {
private:
  std::vector<std::shared_ptr<Neuron>> neurons;

public:
  Layer(const std::vector<std::shared_ptr<Neuron>> &neurons);

  std::vector<std::shared_ptr<Value>>
  operator()(const std::vector<std::shared_ptr<Value>> &input) const;

  std::vector<std::shared_ptr<Value>> parameters() const override;
};

class MLP : public Module {
private:
  std::vector<std::shared_ptr<Layer>> layers;

public:
  MLP(const std::vector<std::shared_ptr<Layer>> &layers);
  std::vector<std::shared_ptr<Value>>
  operator()(const std::vector<std::shared_ptr<Value>> &input) const;

  std::vector<std::shared_ptr<Value>> parameters() const override;
};
