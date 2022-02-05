# End-to-end Customer Segmentation Project

_Note: This project is in progress._

## Tools Used in This Project
* [Prefect](https://www.prefect.io/): Orchestrate workflows
* [hydra](https://hydra.cc/): Manage configuration files
* [pre-commit plugins](https://towardsdatascience.com/4-pre-commit-plugins-to-automate-code-reviewing-and-formatting-in-python-c80c6d2e9f5?sk=2388804fb174d667ee5b680be22b8b1f): Automate code reviewing formatting 
* [poetry](https://python-poetry.org/): Python dependency management
* [DVC](https://dvc.org/): Data version control

## Project Structure
* `src`: consists of Python scripts
* `config`: consists of configuration files
* `notebook`: consists of Jupyter Notebooks
* `tests`: consists of test files

## How to Run the Project
1. Install [Poetry](https://python-poetry.org/docs/#installation)
2. Set up the environment:
```bash
make setup
```
3. Run the process pipeline:
```bash
poetry run python src/process_data.py
```


# This pipeline was automatically generated

## Setup

```sh
pip install -r requirements.txt
```

## Usage

List tasks:

```sh
ploomber status
```

Execute:

```sh
ploomber build
```

Plot:

```sh
ploomber plot
```

*Note:* plotting requires `pygraphviz` for instructions, [see this.](https://docs.ploomber.io/en/latest/user-guide/faq_index.html#plotting-a-pipeline)

## Resources

* [Ploomber documentation](https://docs.ploomber.io)