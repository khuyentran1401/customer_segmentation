# End-to-end Customer Segmentation Project

## Tools Used in This Project
* [hydra](https://hydra.cc/): Manage configuration files - [article](https://towardsdatascience.com/introduction-to-hydra-cc-a-powerful-framework-to-configure-your-data-science-projects-ed65713a53c6)
* [Prefect](https://www.prefect.io/): Orchestrate data workflows - [article](https://towardsdatascience.com/orchestrate-a-data-science-project-in-python-with-prefect-e69c61a49074)
* [Weights & Biases](https://wandb.ai/): Track and monitor experiments - [article](https://towardsdatascience.com/introduction-to-weight-biases-track-and-visualize-your-machine-learning-experiments-in-3-lines-9c9553b0f99d)

## Project Structure for Each Step
* `src`: consists of Python scripts
* `config`: consists of configuration files
* `notebook`: consists of Jupyter Notebooks
* `tests`: consists of test files
* `data`: consists of data

## How to Run the Project
1. Clone this branch:
```bash
git clone --branch workshop https://github.com/khuyentran1401/customer_segmentation.git
```
2. Install [Poetry](https://python-poetry.org/docs/#installation)
3. Set up the environment:
```bash
make setup
```
4. Start with [step0](./step0).
