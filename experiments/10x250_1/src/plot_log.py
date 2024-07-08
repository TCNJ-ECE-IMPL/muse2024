import matplotlib.pyplot as plt
import re
import ast

# Plots a .log function created from network_optuna

###########################
##   Reading .log file   ##
###########################

## Trial [value] finished with value: [decimal value] and parameters: [new dictionary of parameters]
trial_format = re.compile(r"Trial (\d+) finished with value: ([\d\.]+) and parameters: ({.*?}).")

## Modify this to change log path
log_path = "src/study.log"

## Open log
file = open(log_path,"r")

## Skip first line; this can be modified to make use of the study name
first_line = file.readline

## Create trials array
trials = []

## Parse log file and store trial info
for line in file:
    curr_trial = trial_format.search(line)              ## Check if line follows format
    if curr_trial:                                      ## If yes...
        trial_num = int(curr_trial.group(1))                ## Gather trial number, finished value, and parameters
        val = float(curr_trial.group(2))
        param = ast.literal_eval(curr_trial.group(3))

        trial_details = {                               ## Group our info, assign tags
            "trial_num" : trial_num,
            "val" : val,
            "param" : param
        }

        trials.append(trial_details)                    ## Append to trials array

######################
##   Create graph   ##
######################

## Assign axes
x_axis = [trial["trial_num"] for trial in trials]
y_axis = [trial["val"] for trial in trials]

## Plotting 
plt.plot(x_axis, y_axis)
 
## Assign labels and title
plt.xlabel('Trial Number')
plt.ylabel('Value')
plt.title('Test')
 
# Display plot
plt.grid(True, which="both", ls="--")
plt.show()