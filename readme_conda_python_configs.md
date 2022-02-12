WellPlate_project
==============================


conda update -n base -c defaults conda


# how to create developing environment [env]

conda env create -f WellPlate-environment.yml 
NOTE: it may take some time to solve all the rependencies!

conda env remove -n  multiwell-env


conda activate multiwell-env
cd multiwell_code
pip install -e .


conda activate multiwell-env
conda install -c conda-forge spyder 
spyder &


conda install opencv == 3.4.12
conda install tensorflow==1.14
conda install tensorflow-gpu==1.13.1

- "opencv>=4.5" 

channels:
  - defaults
  - conda-forge
  
"opencv-python>=4.4"

"opencv-contrib-python-headless>=4.4"


anaconda-navigator




# on server
conda activate kg-env ; spyder --new-instance &


# Extra
conda update conda
conda update anaconda
conda update python
conda update --all

conda clean --all





# Tips for developers
pip install pep8
pip install pylint


advice in 2020, to use Visual Studio Code

Install  Python support


VS Code Quick Open (Ctrl+P)

ext install ms-python.python



# Tips for VS code
Extension:
Python  (ms-python.python)
GitLens — Git supercharged
Code Spell Checker


# Configs for VS code
File > Preferences > keyboard shortcut
F5: file Python interactive
F9: line Python interactive

File > AutoSave


Press: CTRL + Shift + P

Click on "Preferences: Open Settings (JSON)"

Add this line into JSON : "python.linting.pylintArgs": ["--generate-members"]


#https://stackoverflow.com/questions/56844378/pylint-no-member-issue-but-code-still-works-vscode
  
