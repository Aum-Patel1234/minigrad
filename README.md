# Minigrad

Minigrad is a minimal C++ autograd / computational graph project with Python bindings using **pybind11**.
It includes a `Value` class that supports basic arithmetic (`+`, `-`, `*`, `/`) and can be extended for autograd functionality.

This project is primarily for learning purposes, to understand how autograd and computational graphs work.

Inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy.

## IMPORTANT Notes
When a lambda captures a shared_ptr, it increments the reference count, keeping the object alive. Capturing a weak_ptr does not increase the count, so the object can still be destroyed.
```cpp
std::shared_ptr<Value> a = std::make_shared<Value>(10);
auto cb1 = [a]{ a->grad += 1; };           // captures shared_ptr → ref count +1
std::weak_ptr<Value> wa = a;
auto cb2 = [wa]{ if(auto s = wa.lock()) s->grad += 1; }; // captures weak_ptr → no ref count increase
```

## Prerequisites

Before building the project, make sure you have:

1. **C++ compiler** with C++17 support (`g++` or `clang++`)
2. **CMake** (>= 3.14)
3. **Python** (>= 3.10)
4. **pip** to install Python packages

## Setup Instructions

1. **Clone the repository**:

```bash
git clone https://github.com/Aum-Patel1234/minigrad.git
cd minigrad
```

2. **Install Python dependencies**:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. **Build the pybind11 module**:

```bash
mkdir -p core/build
cd core/build
cmake ..
cmake --build .
```

* This will compile the `minigrad` Python module (`.so` file) in `core/build`.

4. **Generate Python stubs (optional)**:

```bash
cd ../..  # back to project root
bash generate_stubs.sh
```

* This generates a `.pyi` file for autocompletion.

---

## Running the Demo

Run the Python demo script:

```bash
export PYTHONPATH=$(pwd)/core/build
python src/demo.py
```

You should see output like:

```
a = <Value data=2>
b = <Value data=3>
c = a + b = <Value data=5>
d = a - b = <Value data=-1>
e = c * d = <Value data=-5>
f = e / a = <Value data=-2.5>
```

---

## Extending MiniGrad

* You can add gradients tracking in `Value` class.
* More operators and math functions can be added to `Value` class and bound in `bindings.cpp`.
* Use this project as a minimal foundation to build your own autograd engine.