#include "nn.hpp"
#include "value.h"
#include <memory>

Neuron::Neuron(size_t inputSize) {
  static std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<double> dist(-1, 1);

  w.reserve(inputSize);
  for (size_t i = 0; i < inputSize; ++i)
    w.push_back(std::make_shared<Value>(dist(rng)));

  b = std::make_shared<Value>(dist(rng));
}

std::shared_ptr<Value>
Neuron::operator()(const std::vector<std::shared_ptr<Value>> &input) const {
  if (input.size() != w.size())
    throw std::runtime_error("Input and weight size mismatch");

  if (w.size() == 0)
    throw std::runtime_error("size cant be 0.");

  auto out = *w[0] * input[0];
  for (size_t i = 1; i < w.size(); ++i)
    out = *out + (*w[i] * input[i]);

  out = *out + b;
  // TODO: accept activation func as an argument
  return out->tanh();
}

std::vector<std::shared_ptr<Value>> Neuron::parameters() const {
  std::vector<std::shared_ptr<Value>> params(w);
  params.push_back(b);
  return params;
}

// Layers
Layer::Layer(const std::vector<std::shared_ptr<Neuron>> &neurons)
    : neurons(neurons) {};

std::vector<std::shared_ptr<Value>>
Layer::operator()(const std::vector<std::shared_ptr<Value>> &input) const {
  std::vector<std::shared_ptr<Value>> out;
  out.reserve(neurons.size());

  for (const auto &n : neurons)
    out.push_back((*n)(input)); // this willl accept a new neuron

  return out;
}

std::vector<std::shared_ptr<Value>> Layer::parameters() const {
  std::vector<std::shared_ptr<Value>> params;
  for (const auto &p : neurons) {
    auto parameters = p->parameters();
    params.insert(params.end(), parameters.begin(), parameters.end());
  }
  return params;
}

// MLP
MLP::MLP(const std::vector<std::shared_ptr<Layer>> &layers) : layers(layers) {};

std::vector<std::shared_ptr<Value>>
MLP::operator()(const std::vector<std::shared_ptr<Value>> &input) const {
  std::vector<std::shared_ptr<Value>> out = input;

  for (const auto &l : layers) {
    out = (*l)(out);
  }

  return out;
}

std::vector<std::shared_ptr<Value>> MLP::parameters() const {
  std::vector<std::shared_ptr<Value>> out;

  for (const auto &l : layers) {
    auto np = l->parameters();
    out.insert(out.end(), np.begin(), np.end());
  }

  return out;
}
