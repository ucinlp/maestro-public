# Defense War

## Dowloading datasets
Download the datasets from here and copy them the following path
```
tasks/defense_war/datasets/CIFAR10/student/
```

## Assignment
Complete or revise the train.py and predict.py in the following path and then run the train.py file by the following commands to generate the defense_war-model.pth file.
```
$ cd tasks/defense_war/submission/ # predict.py and train.py are under the path.
$ python train.py
```

## Evaluating the submissions
Run the evaluator by following steps
```
$ cd tasks/defense_war/
$ python Evaluator_defense_war.py
```


## Results
View your results at
```
tasks/defense_war/results.json
```

## Evaluation metrics
The score contains two parts: The raw accuracy score cover 40% and four attack methods cover 60% equally. The raw accuracy over 50% starts to earn points and if the rar accuracy is over 77% you can get the whole 40 points. As to each attack method, it covers 15 points. Two attack methods (PGD and one black box attack) will be released and the other two are hidden.

`raw accuracy score = max((min(results["raw_acc"] - 50, 27)/27)*40, 0)`
`each attack method score = (max(100-success_rate, 0)/100) * 70 + (1-max(1500-query, 0)/1500) * 20 + (1-max(15-distance, 0)/15) * 10`

