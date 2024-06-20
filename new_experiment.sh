#!/bin/bash

read -p "Enter name of experiment: " exp_name
exp_dir="./experiments/$exp_name"

while [ -d "$exp_dir" ]; do
    read -p "Already exists, please enter new name: " exp_name
    exp_dir="./experiments/$exp_name"
done

mkdir -p "$exp_dir"
cp -r ./src "$exp_dir/src"
cp -r ./tdms_data "$exp_dir/tdms_data"

echo "Running script..."
cd $exp_dir
python3.10 ./src/network_optuna.py