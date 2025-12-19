#!/bin/bash
set -e # Exit on any error

# Paths
BUILD_DIR="core/build"
STUB_OUT="core/stubs_tmp"
MODULE_NAME="minigrad"

# Make sure build directory exists
if [ ! -d "$BUILD_DIR" ]; then
  echo "ERROR: Build directory '$BUILD_DIR' does not exist!"
  exit 1
fi

mkdir -p "$STUB_OUT"

# Generate pybind11 stub
echo "Generating stub for $MODULE_NAME..."
PYTHONPATH="$BUILD_DIR" pybind11-stubgen "$MODULE_NAME" --output-dir "$STUB_OUT"

if [ ! -f "$STUB_OUT/$MODULE_NAME.pyi" ]; then
  echo "ERROR: Stub file not generated!"
  exit 1
fi

mv "$STUB_OUT/$MODULE_NAME.pyi" "$BUILD_DIR/$MODULE_NAME.pyi"

rm -rf "$STUB_OUT"

echo "✅ Stub generation complete and moved to $BUILD_DIR"
