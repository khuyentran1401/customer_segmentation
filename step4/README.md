# Step 3

## Changes
* Save all outputs in `process_data.py` locally
* Cache outputs of some tasks in `segment.py`.

## Advantages
* Observe the outputs of all tasks by adding one line of code
* Avoid recomputing expensive and time-consuming computations that are unlikely to change 

## Disadvantages
* Difficult to compare the outputs of different experiments
* Cannot reuse the transformers and models of the experiment with the best result

## How to Run Code in This Step
### Run Prefect Locally
To process the data, type:
```bash
python src/process_data.py
```

### Run Prefect Cloud
If you didn't create a project on Prefect, start with creating one:
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
at the end of the files `segment.py`.

Now run:
```bash
python src/segment.py
```
to register the `segment` flow.



