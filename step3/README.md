# Step 3

## Changes
* Add [Prefect](https://www.prefect.io/) to monitor data pipelines

## Advantages
* Understand the relationships between different functions
* Know which functions are being executed
* Understand the duration of one function compared to other functions
* Version the flow, which enables to go back and forth between different versions

## Disadvantages
* Cannot observe the outputs of each task

## How to Run Code in This Step
### Run Prefect Locally
Go to the `step3` directory:
```bash
$ cd step3
```

To process the data, type:
```bash
$ python src/process_data.py
```
To segment the data, type:
```bash
python src/segment.py
```
### Run Prefect Cloud
Start with creating a project on Prefect by running:
```bash
$ prefect create project "customer_segmentation"
```
Next, start a local agent to deploy our flows locally on a single machine:
```bash
$ prefect agent local start
```
Then add:
```python
flow.register(project_name="customer_segmentation")
```
at the end of the files `process_data.py` and `segment.py`.

Now run:
```bash
python src/process_data.py
```
to register the `process_data` flow.

## Next step
Go to [step4](../step4).
