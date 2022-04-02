# maestro-public
ICS175 Student Version for the Spring quarter 2022

## Description

## Creating Environment
Make sure you have the virtual environment set (at the root folder). You can use any virtual environment, for instance [venv](https://docs.python.org/3/tutorial/venv.html) or [anaconda](https://docs.anaconda.com/anaconda/install/index.html). Example:
```
$ python3 -m venv env
$ source env/bin/activate
```
Once the virtual environment is set, install the requirements:
```
$ (env) pip3 install -r requirements.txt
```
Finally, install Maestro:
```
$ (env) python3 -m pip install -e .
```

## Dowloading datasets
Download the datasets from here and copy them the following path 
```
tasks/attack_homework/datasets/MNIST/student/
```

## Assignment
Complete the missing code in ToDo sections of attack.py in the following path
```
tasks/attack_homework/submission/attack.py
```

## Evaluating the submissions
Run the evaluator by following steps
'''
cd tasks/attack_homework/
python Evaluator_attack_homework.py
'''

## Results
View your results at 
```
tasks/attack_homework/results.txt
```
