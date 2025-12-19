#pragma once

#include <ostream>
class Value {
private:
  double data;

public:
  Value(double data);
  double getData() const;

  Value operator+(const Value &other) const;
  Value operator-(const Value &other) const;
  Value operator*(const Value &other) const;
  Value operator/(const Value &other) const;

  // friend allows a function or class to access private and protected members
  friend std::ostream &operator<<(std::ostream &os, const Value &val);
};
