# How to run

1. Copy the contents of the intended dataset to be used to the /tdms_data/ directory. Copies of different datasets which can be used are stored in the /datasets/ directory.

2. Run new_experiment.sh, which will prompt for a name for the experiment. This will copy the files in /src/ and /tdms_data/, as well as any new files needed to run a new test. prepare_dataset.py will automatically be run.

3. The new experiment will be in the /experiments/ directory. Any changes to the source code within these directories that you intend to keep in following experiments should be mirrored in the base /src/ directory. Always execute the python scripts created for the experiment directory, not the original source copy.

4. Run network_optuna.py to create an optuna study. Once completed, a model will be created under /models/optuna_best.keras which will serve as a good starting point for the network. Full details are contained in /src/study.log

5. Either train the optuna_best.keras model using train_model.py, or re-create the network in network.py utilizing the produced log.