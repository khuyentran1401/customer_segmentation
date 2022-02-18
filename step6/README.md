# Step 3

## Changes
* Add [Weights & Biases](https://wandb.ai) to track and monitor experiments

## Advantages
* Log all experiments
* Easy to compare between different experiments
* Easy to reproduce a particular experiment

## How to Run Code in This Step
Go to `step5` directory:
```bash
$ cd step5
```
Run the entire project using the configuration `process_1`:
```bash
$ python src/main.py process=process_1
```

Run the entire project using the configuration `process_2`:
```bash
$ python src/main.py process=process_2
```

Run the entire project using the configuration `process_3`:
```bash
$ python src/main.py process=process_3
```

Run the entire project using the configuration `process_3`:
```bash
$ python src/main.py process=process_4
```

After comparing the results of different experiments, we choose the best versions of the models for the file `predict.py`, then run:

```python
$ python src/predict.py
```