#include "nn.hpp"
#include "value.h"
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(minigrad, m) {
  m.doc() = "MiniGrad Value class Python bindings";

  py::class_<Value, std::shared_ptr<Value>>(m, "Value")
      .def(py::init<double>(), "Constructor with a double value")
      .def("getData", &Value::getData, "Get the internal data")
      // Overload + operator
      .def("__add__", [](std::shared_ptr<Value> a,
                         std::shared_ptr<Value> b) { return *a + b; })
      .def("__sub__", [](std::shared_ptr<Value> a,
                         std::shared_ptr<Value> b) { return *a - b; })
      .def("__mul__", [](std::shared_ptr<Value> a,
                         std::shared_ptr<Value> b) { return *a * b; })
      .def("__truediv__", [](std::shared_ptr<Value> a,
                             std::shared_ptr<Value> b) { return *a / b; })
      .def("setData", &Value::setData, "Set the internal data")
      .def("tanh", &Value::tanh)
      .def("relu", &Value::relu)
      .def("pow",
           static_cast<std::shared_ptr<Value> (Value::*)(int)>(&Value::pow),
           py::arg("n"))
      .def("getOp", &Value::getOp)
      .def("getGrad", &Value::getGrad)
      .def("getLabel", &Value::getLabel)
      .def("getRefCount", &Value::getRefCount)
      // Backpropagation
      .def("backPropogate", &Value::backPropogate,
           "Perform backward pass to compute gradients")
      // Optional: define __repr__ for nice printing in Python
      .def("__repr__",
           [](const Value &v) {
             std::string label =
                 std::string(v.getLabel()); // convert string_view -> string
             if (label.empty()) {
               label = "Value(" + std::to_string(v.getData()) + ")";
             } else {
               label += "\\nValue(" + std::to_string(v.getData()) + ")";
             }
             return "<" + label + ">";
           })
      .def("exportGraph", [](const std::shared_ptr<Value> &v) {
        py::dict result;

        std::vector<std::shared_ptr<Value>> nodes;
        std::vector<std::pair<std::shared_ptr<Value>, std::shared_ptr<Value>>>
            edges;
        v->buildGraph(nodes, edges);

        result["nodes"] = nodes;
        result["edges"] = edges;

        return result;
      });

  py::class_<Neuron, std::shared_ptr<Neuron>>(m, "Neuron")
      .def(py::init<size_t>(), py::arg("input_size"),
           "Create a neuron with given input size")
      .def(
          "__call__",
          [](std::shared_ptr<Neuron> &n,
             std::vector<std::shared_ptr<Value>> &inputs) {
            return (*n)(inputs); // call operator()(...) const
          },
          py::arg("inputs"))
      .def("zero_grad", &Neuron::zero_grad);

  // Layer now accepts vector<shared_ptr<Neuron>>
  py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer")
      .def(py::init<const std::vector<std::shared_ptr<Neuron>> &>(),
           py::arg("neurons"))
      .def(
          "__call__",
          [](std::shared_ptr<Layer> &l,
             std::vector<std::shared_ptr<Value>> &inputs) {
            return (*l)(inputs);
          },
          py::arg("inputs"))
      .def("parameters", &Layer::parameters);

  // MLP with shared_ptr<Layer>
  py::class_<MLP, std::shared_ptr<MLP>>(m, "MLP")
      .def(py::init<const std::vector<std::shared_ptr<Layer>> &>(),
           py::arg("layers"))
      .def(
          "__call__",
          [](std::shared_ptr<MLP> &m_,
             std::vector<std::shared_ptr<Value>> &inputs) {
            return (*m_)(inputs);
          },
          py::arg("inputs"))
      .def("parameters", &MLP::parameters)
      .def("zero_grad", &MLP::zero_grad);
}
