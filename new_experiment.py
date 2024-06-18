import os
import shutil

exp_name = input("Enter name of experiment: ")
exp_dir = os.path.join("./experiments", exp_name)

while os.path.exists(exp_dir):
    exp_name = input("Already exists, please enter new name: ")
    exp_dir = os.path.join("./experiments", exp_name)

os.mkdir(exp_dir)
shutil.copytree("./src", os.path.join(exp_dir, "./src"))
shutil.copytree("./tdms_data", os.path.join(exp_dir, "./tdms_data"))
