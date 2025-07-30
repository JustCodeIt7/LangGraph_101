.PHONY: install test lint format clean build dist

# Install the package in development mode
install:
	pip install -e .[dev]

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-coverage:
	pytest tests/ --cov=repository_analyzer --cov-report=html

# Run linting
lint:
	mypy repository_analyzer/
	black --check repository_analyzer/ tests/
	isort --check-only repository_analyzer/ tests/

# Format code
format:
	black repository_analyzer/ tests/
	isort repository_analyzer/ tests/

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/

# Build distribution
build:
	python -m build

# Upload to PyPI (requires twine)
dist:
	python -m build
	twine upload dist/*

# Run pre-commit hooks
pre-commit:
	pre-commit run --all-files

# Install pre-commit hooks
pre-commit-install:
	pre-commit install

# Run all checks
check: lint test

# Help
help:
	@echo "Available targets:"
	@echo "  install        - Install the package in development mode"
	@echo "  test           - Run tests"
	@echo "  test-coverage  - Run tests with coverage"
	@echo "  lint           - Run linting"
	@echo "  format         - Format code"
	@echo "  clean          - Clean build artifacts"
	@echo "  build          - Build distribution"
	@echo "  dist           - Build and upload to PyPI"
	@echo "  pre-commit     - Run pre-commit hooks"
	@echo "  pre-commit-install - Install pre-commit hooks"
	@echo "  check          - Run all checks"
	@echo "  help           - Show this help"