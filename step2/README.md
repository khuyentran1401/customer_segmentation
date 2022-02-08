# Step 2

## Changes
* Create functions for code 
* Turn values that are frequently changed into parameters of functions

## Advantages
* Easier to experiment with different values

## Disadvantages
* It is hard for new users to have a general understanding of the process
* Need to dig into the source code to change values of the parameters


## Things Need to be Improved
* Find a way to visualize the connections between functions
* Use configuration files to collect all parameters into one place

## How to Run the Code
Follow [this instruction](https://docs.prefect.io/orchestration/getting-started/set-up.html#server-or-cloud) to install relevant dependencies for Prefect Cloud.

To create a project on Prefect Cloud, in the main directory, type:
```bash
$ prefect create project 'Customer segmentation'
```

Next, start a local agent to deploy our flows locally on a single machine:
```bash
$ prefect agent local start
```