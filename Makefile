.PHONY: notebook

install: 
	@echo "Installing..."
	poetry install

activate:
	@echo "Activating virtual environment"
	poetry shell

pull_data:
	@echo "Pulling data..."
	poetry run dvc pull

setup: activate 
install_all: install pull_data 

test:
	pytest

clean: 
	@echo "Deleting log files..."
	find . -name "*.log" -type f -not -path "./wandb/*" -delete