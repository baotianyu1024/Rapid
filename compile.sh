#!/bin/bash
# Compile all C++ tools
set -e
cd "$(dirname "$0")"

echo "Compiling CPS tools..."
g++ -O3 -std=c++17 -fopenmp -pthread cps/arrange_ids_to_devices.cpp -o cps/arrange_ids_to_devices
g++ -O3 -std=c++17 -fopenmp -pthread cps/generate_local_dataset.cpp -o cps/generate_local_dataset

echo "Compiling analysis tools..."
g++ -O3 -std=c++17 -fopenmp -pthread analysis/verify_hit_rate.cpp -o analysis/verify_hit_rate

echo "Compiling evaluation tools..."
g++ -O3 -std=c++17 -fopenmp -pthread evaluation/evaluate_partition.cpp -o evaluation/evaluate_partition
g++ -O3 -std=c++17 -fopenmp -pthread evaluation/evaluate_with_sample_assignment.cpp -o evaluation/evaluate_with_sample_assignment

echo "All compiled successfully."
