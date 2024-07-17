#!/bin/bash

EXPERIMENTS_FOLDER_DIR="./experiments_1___2_Class_Classification"
PREP_DATASET="./prepare_dataset.py"

read -p "Enter name of experiment: " exp_name
exp_dir="$EXPERIMENTS_FOLDER_DIR/$exp_name"

while [ -d "$exp_dir" ]; do
    read -p "Already exists, please enter new name: " exp_name
    exp_dir="$EXPERIMENTS_FOLDER_DIR/$exp_name"
done

mkdir -p "$exp_dir"
cp -r ./tdms_data "$exp_dir/tdms_data"
mkdir -p "$exp_dir/src"
mkdir -p "$exp_dir/models"
mkdir -p "$exp_dir/checkpoints"
mkdir -p "$exp_dir/datasets"
cp ./src/2_Class_Classification/prepare_dataset.py "$exp_dir/src/prepare_dataset.py"
cp ./src/2_Class_Classification/network_functions.py "$exp_dir/src/network_functions.py"
cp ./src/2_Class_Classification/network_optuna.py "$exp_dir/src/network_optuna.py"
cp ./src/2_Class_Classification/plot.py "$exp_dir/src/plot.py"
cp ./src/2_Class_Classification/train_model.py "$exp_dir/src/train_model.py"

echo "Running script..."
cd $exp_dir/src
python3.10 "$PREP_DATASET"