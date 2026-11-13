# Predator

AI Lab university project (BS CS 5-A). Predicts whether an online shopping session ends in a purchase (UCI Online Shoppers Purchasing Intention dataset) and explains which features drive the prediction.

## Stack

Python 3.15+ (deps in `pyproject.toml`, locked in `uv.lock`; `uv sync`), pandas, scikit-learn, XGBoost (benchmark only), matplotlib/seaborn, joblib, Streamlit, pytest. No deep learning, no synthetic data.

## Layout

- `predator/` the package: `data.py`, `preprocess.py`, `models.py`, `train.py`, `evaluate.py`, `significance.py`, `curves.py`, `explain.py`, `calibration.py`, `threshold.py`, `errors.py`, `predict.py`
- `main.py` runs the full pipeline; `app.py` is the Streamlit demo
- `notebooks/predator_walkthrough.ipynb` walkthrough that imports the package
- `kaggle/predator-kaggle.ipynb` self-contained copy of the pipeline for Kaggle (no package imports)
- `outputs/` generated figures and CSV tables (committed), `models/` saved models, model_info.json and out-of-fold probabilities (committed for the hosted app; re-run `main.py` and commit after dependency upgrades), `data/` dataset (ignored)
- `docs/` methodology, algorithms, business threshold, error analysis, code guide, results, model card, deploy, Kaggle publishing

## Commands

- Full pipeline: `python main.py` (needs `data/online_shoppers_intention.csv`)
- Demo app: `streamlit run app.py`
- Prediction: `python -m predator.predict examples/session_buyer.json`
- Tests: `pytest -q`; lint: `ruff check . && ruff format --check .`
- Rebuild the Kaggle notebook after changing package code: `python scripts/build_notebooks.py`

## Conventions

- Code is written in a beginner-student style on purpose: plain functions, descriptive names, docstrings on every function, comments explaining the AI concepts. Keep it that way.
- `random_state=42` everywhere. Preprocessing is fitted inside the Pipeline on training data only.
- XGBoost is always labelled as a benchmark beyond the course.
- Checks: after code changes run `pytest -q`; after pipeline changes run `python main.py` and refresh `docs/results.md` and the README results table.
