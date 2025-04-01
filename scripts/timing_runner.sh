#!/bin/bash

n_clients=(2 4 8)

for n_client in "${n_clients[@]}"
do
    echo "Running experiment with $n_client clients"
    mpirun -np $((n_client+1)) python3 experiments.py --exp_type "timing"
done