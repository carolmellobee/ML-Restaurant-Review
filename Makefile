.PHONY: install data data-llm notebooks test robustness all clean

install:
	pip install -r requirements.txt

## data: offline reviews + simulated experiment (no API key needed)
data:
	python scripts/generate_reviews_offline.py
	python -m src.experiment.simulate

## data-llm: reviews written by Claude (needs ANTHROPIC_API_KEY)
data-llm:
	python scripts/generate_reviews_llm.py
	python -m src.experiment.simulate

robustness:
	python scripts/seed_robustness.py

notebooks:
	cd notebooks && for nb in 01_review_nlp 02_power_analysis 03_ab_test_analysis 04_traps_and_recommendation; do \
		jupyter nbconvert --to notebook --execute --inplace $$nb.ipynb; done

test:
	pytest -q

all: data robustness notebooks test

clean:
	rm -rf data/processed/* reports/figures/*
