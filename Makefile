.PHONY: install dev-install test test-fast test-theory diagnostic protocol compare dilution ablation figures full-pipeline clean lint format typecheck

# Setup
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

# Testing
test:
	pytest -v

test-fast:
	pytest -x -q

test-theory:
	pytest spectral_distillation/tests/test_theoretical_bounds.py -v

# Experiments (see SPECTRAL_DISTILLATION_IMPLEMENTATION_GUIDE.md §9)
diagnostic:
	python -m spectral_distillation.experiments.run_diagnostic

protocol:
	python -m spectral_distillation.experiments.run_protocol

compare:
	python -m spectral_distillation.experiments.compare_methods

dilution:
	python -m spectral_distillation.experiments.dilution_curve

ablation:
	python -m spectral_distillation.experiments.ablation

figures:
	python -m spectral_distillation.experiments.reproduce_figures

full-pipeline: diagnostic protocol compare dilution ablation figures

# Utilities
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -name "*.pyc" -delete

lint:
	ruff check .

format:
	ruff format .

typecheck:
	ty check .