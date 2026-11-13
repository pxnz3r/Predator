# Deploying the app

The Streamlit app runs for free on [Streamlit Community Cloud](https://streamlit.io/cloud). Everything it needs is in the repo:

- `models/`: the trained pipelines, `model_info.json` and the out-of-fold probabilities the threshold re-tuning uses. They are committed, so the hosted app does not train anything.
- `outputs/`: the tables and figures shown on the comparison tab.
- `uv.lock`: Community Cloud installs the exact locked versions with uv. Pickled scikit-learn models only load reliably with the version that saved them, so keep the lock and the models in sync: after `uv lock --upgrade`, run `python main.py` and commit `models/` again.

The dataset is not needed at run time.

## Steps

1. Push the repo to GitHub (public, or private with Community Cloud access granted).
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Click **Create app**, then **Deploy a public app from GitHub**.
4. Fill in:
   - Repository: `YOUR_GITHUB_USERNAME/Predator`
   - Branch: `main` (or the branch you publish)
   - Main file path: `app.py`
   - App URL: pick a subdomain, e.g. `predator-purchase`
5. Open **Advanced settings** and set **Python version** to **3.15** (the project needs 3.15 or newer). No secrets are needed.
6. Click **Deploy**. The first build takes a few minutes while uv installs the dependencies.
7. Put the app URL in the README (the "Live demo" link at the top) and in the Kaggle notebook's links.

## Updating

Every push to the deployed branch redeploys the app automatically. After changing the pipeline, run `python main.py` and commit `models/` and `outputs/` together so the app shows matching models and figures.

## Troubleshooting

| Problem | Fix |
|---|---|
| "No trained models found in models/" | `models/` was not pushed. Run `python main.py` and commit `models/` |
| Build fails resolving Python | Python version in Advanced settings is below 3.15. Delete the app and redeploy with 3.15 |
| Error loading a `.joblib` file | The scikit-learn or XGBoost version differs from the one that saved it. Re-run `python main.py` with the locked versions (`uv sync`) and commit `models/` |
| App sleeps after a few days | Free apps sleep when unused; the first visitor wakes it in about 30 seconds |
