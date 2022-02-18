# Step 2

## Changes
* Replace hard-coding values with a config file
* Use [Hydra](http://hydra.cc/) to manage the config file 
## Advantages
* All values are in one place
* Easier to experiment with different parameters 
* Can override the default values from your command line
* Can create multiple config files of the same type
## Disadvantages
* Difficult to understand the relationships between functions
* Difficult to know which functions are being executed
* Cannot check the outputs of a function

## How to Run Code in This Step
Go to the `step2` directory:
```bash
$ cd step1
```
To process the data using the default configurations, type:
```bash
$ python src/process_data.py
```

To process the data using the configuration `process_1`, type:
```bash
$ python src/process_data.py process=process_1
```