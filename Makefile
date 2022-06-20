test:
	pytest

clean: 
	@echo "Deleting log files..."
	find . -name "*.log" -type f -not -path "./wandb/*" -delete