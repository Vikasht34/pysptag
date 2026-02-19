#!/bin/bash
# Build Rust extension and install

cd rust_io

# Build release version
cargo build --release

# Copy to Python package
cp target/release/libfast_io.so ../src/io/fast_io.so || \
cp target/release/libfast_io.dylib ../src/io/fast_io.so

echo "Built Rust extension successfully!"
