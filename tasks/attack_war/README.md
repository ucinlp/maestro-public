# Attack War_Phase

## Description
To be updated

## Dowloading datasets
Download the datasets from here and copy them the following path 
```
tasks/attack_war/datasets/CIFAR10/student/
```

## Assignment
Write your attack method in ToDo sections of attack.py in the following path
```
tasks/attack_war/submission/attack.py
```
You could also add other files/models like detector model in defense_project and use it in the code. These folders need to be present in the following folder
```
tasks/attack_war/submission/
```
And you can load these files in you attack.py using attack_path argument passed in attack class instance.

## Evaluating the submissions
Run the evaluator by following steps
```
$ cd tasks/attack_war/
$ python Evaluator_attack_project.py
```

## What to submit
You need to submit attack.py and all the files/models you used for attack.py if any (only as files of submission folder NOT the submission folder)

## Results
View your results at 
```
tasks/attack_war/results.json
```

## Evaluation metrics
You will be evaluated on 3 defense models, 2 given to you 1 hidden. For each defense model the score is calculated as below.
A score is generated based on your results(attack success rate, queries, distance). Score is calculated as follows:
```
Score = 70* max(success_rate - 40, 0)/60 +  20 * max(1000 - total_queries, 0)/1000 + 10 * max(15 - distance, 0)/15
```
This is tentative and may change based on submissions.

## Things to be taken note of:
1. get_batch_output function is changed which will output a tuple - (output, detected). 
2. You can load/add extra files/models for your attack.py
3. Currently there is only one defender is made available to you, second model will be updated in a couple of days. 

# GOOD LUCK !!!
