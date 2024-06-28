# How to run

1. Copy the contents of the intended dataset to be used to the /tdms_data/ directory

2. Run new_experiment.sh, which will prompt for a name for the experiment. This will copy the files in /src/ and /tdms_data/, as well as any new files needed to run a new test. Optuna will automatically be run.

3. The new experiment will be in the /experiments/ directory. Any changes to the source code within these directories that you intend to keep in following experiments should be mirrored in the base /src/ directory.