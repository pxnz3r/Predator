# Publishing the notebook on Kaggle

The Kaggle version is `kaggle/predator-kaggle.ipynb`. It is self-contained: the code from every `predator/` module is pasted into its cells, and `find_dataset()` finds the CSV under `/kaggle/input/`. No package install or upload of other files is needed.

Before publishing, replace the Kaggle placeholder:

| Where | Placeholder | Replace with |
|---|---|---|
| `kaggle/kernel-metadata.json` | `YOUR_KAGGLE_USERNAME` in `id` | Your Kaggle username |
| `README.md`, `scripts/build_notebooks.py` | `YOUR_KAGGLE_USERNAME` | Your Kaggle username |

Then rebuild the notebook so the link is in it: `python scripts/build_notebooks.py`.

The notebook uses pandas, scikit-learn, XGBoost, matplotlib and seaborn, all preinstalled on Kaggle. Internet access is not needed. A full run takes about 5 minutes on Kaggle's CPU.

## Option A: through the Kaggle website

1. Sign in at [kaggle.com](https://www.kaggle.com) and click **Create > New Notebook**.
2. **File > Import Notebook**, choose `kaggle/predator-kaggle.ipynb` from your computer, and confirm.
3. In the right panel, click **Add Input** (or **Add Data**), search for **Online Shoppers Purchasing Intention Dataset** (`henrysue/online-shoppers-intention`) and add it. It is mounted under `/kaggle/input/`.
4. Check **Session options**: Accelerator **None**, Internet **off** (not needed).
5. Click **Run All** and wait for every cell to finish without errors.
6. Rename the notebook at the top: *Predator: Customer Purchase Behavior Prediction*.
7. Click **Save Version**, choose **Save & Run All (Commit)**, and wait for the version to finish.
8. Open the notebook's page, click **Share**, and set it to **Public**.
9. Copy the notebook's URL into the README's Kaggle link.

## Option B: with the Kaggle CLI

1. Install the CLI: `pip install kaggle`.
2. Create an API token: on kaggle.com go to **Settings > API > Create New Token**. This downloads `kaggle.json`.
3. Put the token where the CLI looks for it, and keep it private (it is in `.gitignore`; never commit it):
   - Linux / macOS: `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
   - Windows: move it to `C:\Users\<you>\.kaggle\kaggle.json`
4. Edit `kaggle/kernel-metadata.json`:
   - `id`: `your-kaggle-username/predator-purchase-prediction`
   - `is_private`: keep `"true"` for the first push so you can check the run, then change it to `"false"` to publish
   - `dataset_sources` already lists `henrysue/online-shoppers-intention`
5. Push from the project folder:

   ```bash
   kaggle kernels push -p kaggle
   ```

   This uploads the notebook, attaches the dataset and starts a run on Kaggle.
6. Check the run:

   ```bash
   kaggle kernels status your-kaggle-username/predator-purchase-prediction
   ```

   Wait until it says `complete`. If it says `error`, download the log with `kaggle kernels output your-kaggle-username/predator-purchase-prediction -p kaggle/output` and look at the last cell that ran.
7. To publish publicly, set `"is_private": "false"` in `kernel-metadata.json` and push again. Each push creates a new version.

## Updating later

After changing code in `predator/`:

```bash
python scripts/build_notebooks.py
kaggle kernels push -p kaggle
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not find the dataset file 'online_shoppers_intention.csv'` | The dataset is not attached. Add it with **Add Input** (website) or check `dataset_sources` (CLI) |
| `401 Unauthorized` from the CLI | `kaggle.json` is missing or in the wrong folder (step 3) |
| `403` on push | The `id` username does not match the account in `kaggle.json` |
| Notebook slug already exists | Change the part after `/` in `id` |
