#pragma once

#include "value.h"
#include <cstddef>
#include <memory>
#include <vector>

class Neuron {
private:
  std::vector<std::shared_ptr<Value>> w;
  std::shared_ptr<Value> b;

public:
  Neuron(size_t inputSize);
  std::shared_ptr<Value> operator()(std::vector<std::shared_ptr<Value>> &input);
};
class Layer {
public:
};

class MLP {
public:
};
