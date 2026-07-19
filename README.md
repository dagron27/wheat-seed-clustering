# Wheat Seed Variety Clustering

![CI](https://github.com/dagron27/se412-clustering/actions/workflows/ci.yml/badge.svg)

**Assignment:** `se412-clustering-main` — SE 412.

## Overview

This project applies unsupervised clustering to the UCI **seeds** dataset
(`dataset/seeds_dataset.csv`: 210 wheat kernels, 7 geometric features, 3
known varieties -- Kama, Rosa, Canadian, 70 kernels each, encoded as
`class` 1/2/3) to see how well each algorithm recovers the true variety
labels without being told what they are. The 8 columns are `area`,
`perimeter`, `compactness`, `kernel_length`, `kernel_width`,
`asymmetry_coef`, `kernel_groove`, and `class`. The work is split across
two independent, self-contained Jupyter notebooks:

- **`seed_analysis_k_means.ipynb`** (28 cells) -- K-Means clustering. Loads
  `dataset/seeds_dataset.csv`, does exploratory data analysis (`.head()`,
  `.info()`, per-class `.describe()`, KDE distribution plots), removes
  per-class outliers via a z-score threshold (cell 15), computes a Pearson
  correlation heatmap, standardizes features, and reduces dimensionality
  with PCA (comparing 2/3/4/5 components in cell 18, plus a 95%-variance
  threshold in cell 17). Fits `KMeans(n_clusters=3)` on the scaled/
  PCA(4)-reduced data via a `create_clustering_pipeline` helper (cell 20),
  scores results with silhouette score and adjusted Rand index (ARI)
  against the known labels, runs a small grid search over `n_init` /
  `max_iter` / `tol` (cell 24), and visualizes clusters with t-SNE (cell
  22) and a 3D Plotly scatter (cell 23; there is no separate 2D Plotly
  view, only `px.scatter_3d`). On the last saved run, the
  4-component pipeline scored silhouette 0.408 / ARI 0.809, and the
  hyperparameter search's best silhouette (0.410) and best ARI (0.796)
  both landed on `n_init=10, max_iter=100, tol=1e-4`.

- **`seed_analysis_dbscan.ipynb`** (49 cells) -- DBSCAN clustering. Same
  dataset load and EDA pattern (head/info/value_counts, per-class
  describe, KDE plots, per-class box plots for outlier inspection,
  correlation table). Builds a
  `Pipeline(StandardScaler -> PCA(n_components=3) -> DBSCAN)`, evaluates
  with silhouette score and ARI, then runs `GridSearchCV` over `eps` /
  `min_samples` to tune DBSCAN (cells 38-40), and visualizes the tuned
  result via PCA scatter and t-SNE. On the last saved run, the untuned
  default (`eps=0.5, min_samples=7`) scored silhouette -0.214 / ARI 0.007;
  after the grid search "tuning" step (see Known Issues -- Correctness
  Notes below, its selection is not trustworthy) the reported best params
  were `eps=0.7, min_samples=4` with ARI 0.274. DBSCAN clearly
  under-performs K-Means on this dataset in both the tuned and untuned
  case.

Both notebooks show contiguous, ascending `execution_count` values with no
error outputs (K-Means 1-27 with no gaps; DBSCAN 1-34 contiguous, with the
final two plotting cells at 38/40 from a couple of extra in-place re-runs of
just those cells), consistent with a full top-to-bottom run each. Their
embedded kernelspec metadata (`nbformat` 4) both name a `venv` / Python
3.12.7 kernel, but this only reliably describes `seed_analysis_k_means.ipynb`.
`seed_analysis_dbscan.ipynb` carries a `"colab"` metadata block
(`provenance`/`include_colab_link`) not present in the K-Means notebook, and
one of its cached warning outputs (cell 46) references
`/usr/local/lib/python3.10/dist-packages/sklearn`, the characteristic path
and Python version of a Google Colab runtime -- not a local Python 3.12.7
venv. This matches the git history: every commit that substantively edited
`seed_analysis_dbscan.ipynb` is titled "Created using Colab" (see
Contributions below). Treat the DBSCAN notebook's saved outputs as having
last been produced in Colab (Python 3.10), not locally, despite what its
kernelspec metadata states.

## Dependencies

A `requirements.txt` is now included (`pip install -r requirements.txt`).
The following are inferred from the notebooks' `import` statements:

- `pandas`
- `numpy`
- `scikit-learn` (`sklearn.cluster`, `sklearn.preprocessing`,
  `sklearn.decomposition`, `sklearn.manifold`, `sklearn.metrics`,
  `sklearn.pipeline`, `sklearn.model_selection`)
- `matplotlib`
- `seaborn`
- `plotly` (K-Means notebook only, for the interactive 3D scatter)
- `packaging` (used only to assert a minimum `scikit-learn` version)

## Environment Setup

1. Create and activate a Python 3 virtual environment (the notebooks'
   embedded kernel metadata names it `venv`, last run on Python 3.12.7).
2. Install the packages listed under Dependencies, e.g.:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn plotly packaging
   ```
3. Launch Jupyter (`jupyter notebook` or `jupyter lab`) from the repository
   root so the notebooks' relative path `dataset/seeds_dataset.csv`
   resolves correctly.
4. Run either notebook top-to-bottom. No API keys, credentials, or network
   access are required.

## Continuous Integration

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push
(and can also be triggered manually via `workflow_dispatch`). It installs
`requirements.txt` plus `jupyter`/`nbconvert`, then re-executes both
notebooks top-to-bottom from scratch with
`jupyter nbconvert --to notebook --execute`, writing the executed output to
a throwaway path under `/tmp` rather than back into the source `.ipynb`
files. The job fails if either notebook raises an error during execution.

A couple of things to know about this:

- **The committed notebooks are never modified by CI.** The execute step
  intentionally omits `--inplace` and instead writes to a separate output
  file that is discarded when the runner is torn down. GitHub-hosted
  runners are also ephemeral (a fresh VM per run) and the workflow never
  commits or pushes anything, so a CI run cannot alter the `.ipynb` files
  in the repository under any circumstances.
- **Re-executed metrics/plots may differ slightly from the committed
  output, and that is expected.** Every `KMeans(...)` call in
  `seed_analysis_k_means.ipynb` and every `TSNE(...)` call in both
  notebooks explicitly passes `random_state=42`, and DBSCAN itself has no
  random component (`DBSCAN(...)` takes no `random_state` parameter), so
  in principle a re-run on the same library versions should reproduce the
  same clusters. In practice, small floating-point differences can still
  show up between environments (e.g. different `scikit-learn` versions or
  BLAS/LAPACK backends between your local machine and the CI runner) and
  would show up as minor changes to silhouette score, ARI, or plot
  rendering rather than a hard failure. Separately, the DBSCAN
  notebook's grid search has a known scoring bug (see Known Issues --
  Correctness Notes, item 6) that makes its "best" hyperparameter
  selection effectively arbitrary; that non-determinism comes from the
  bug itself, not from CI re-execution, and would reproduce locally too.
  None of this affects whether the CI job passes -- it only fails on an
  actual exception raised while executing a cell.

## Known Issues

### Dead Code

1. **`dataset/converter.py`** -- A standalone TSV-to-CSV conversion
   utility (`convert_tsv_to_csv`) with a module-level "Example usage" call
   that runs `seeds_dataset.tsv` -> `seeds_dataset.csv` on import. It is
   never imported or referenced by either notebook; both notebooks read
   `dataset/seeds_dataset.csv` directly, which is already committed to the
   repo.
   - *Fix-it plan*: Either delete the script if the CSV is considered the
     canonical artifact and regeneration is never needed, or convert it
     into a guarded, documented one-off script (e.g. gate the conversion
     call behind `if __name__ == "__main__":`) and reference it from the
     README as the documented way to regenerate the CSV from the TSV
     source.

2. **Commented-out Google Colab drive-mount block** -- present in both
   notebooks (`seed_analysis_k_means.ipynb`, cell 2;
   `seed_analysis_dbscan.ipynb`, cell 3), each reading:
   ```python
   #from google.colab import drive

   ## Mount Google Drive (for Colab users)
   #try:
   #    drive.mount('/content/drive')
   #except:
   #    pass
   ```
   It is inert (fully commented out) and references only the generic
   Colab path `/content/drive`; no personal account or path is embedded.
   - *Fix-it plan*: Remove the dead block, or replace it with a live,
     minimal conditional mount (guarded by an `IN_COLAB` check) if Colab
     execution is still an intended use case; otherwise delete it since
     both notebooks already load data from the local `dataset/` folder.

3. **Unfinished `predict_cluster` helper and usage example** --
   `seed_analysis_k_means.ipynb`, cells 25-26. Cell 25 is a fully
   commented-out function definition:
   ```python
   # def predict_cluster(new_data, pipeline):
   #     scaled_data = pipeline['scaler'].transform(new_data)
   #     pca_data = pipeline['pca'].transform(scaled_data)
   #     predictions = pipeline['kmeans'].predict(pca_data)

   #     return predictions
   ```
   Cell 26 is an unexecuted usage-example string (a bare triple-quoted
   string, not a comment, so it evaluates to a no-op string literal at
   runtime) demonstrating how `predict_cluster` would be called against
   the `results` dict returned by `create_clustering_pipeline` (cell 20).
   This function is specific to the K-Means notebook; the DBSCAN notebook
   has no equivalent.
   - *Fix-it plan*: Either implement and exercise `predict_cluster` against
     a small held-out sample to demonstrate inference on new kernels, or
     remove both cells if out-of-sample prediction is out of scope for the
     assignment.

4. **Commented-out `joblib` model-saving block** --
   `seed_analysis_k_means.ipynb`, cell 27:
   ```python
   # Optional: Save the models for later use
   # import joblib

   # def save_pipeline(pipeline, filename_prefix='cluster_pipeline'):
   #     joblib.dump(pipeline['scaler'], f'{filename_prefix}_scaler.pkl')
   #     joblib.dump(pipeline['pca'], f'{filename_prefix}_pca.pkl')
   #     joblib.dump(pipeline['kmeans'], f'{filename_prefix}_kmeans.pkl')

   # save_pipeline(results)
   ```
   Fully inert. There is no corresponding `joblib.load` anywhere in either
   notebook, so this block carries no deserialization risk as written.
   - *Fix-it plan*: Remove it if model persistence isn't needed, or
     uncomment and wire it up if a saved pipeline is meant to back the
     `predict_cluster` helper above. `*.pkl` is already listed in
     `.gitignore`, so no additional gitignore change would be needed to
     do this safely.

5. **Copy-paste bug in DBSCAN box-plot cells** --
   `seed_analysis_dbscan.ipynb`, cell 18 duplicates cell 16
   (`plt.boxplot(class_dfs[0])`) instead of plotting the second class. The
   surrounding structure is:
   - Cell 16: `plt.boxplot(class_dfs[0])` -> followed by markdown cell 17,
     "For data group 1, outliers exist in: ..."
   - Cell 18: `plt.boxplot(class_dfs[0])` (should be `class_dfs[1]`) ->
     followed by markdown cell 19, "For data group 2, outliers are in
     columns: ..."
   - Cell 20: `plt.boxplot(class_dfs[2])` -> followed by markdown cell 21,
     "data group 3, outlier exists in features: ..."

   As a result, the notebook never actually renders a box plot for class 2
   (`class_dfs[1]`); the written commentary for "data group 2" describes a
   plot that was never generated (it's a re-render of class 1's plot). The
   two markdown cells (17 and 19) currently list the same outlier columns,
   which is consistent with them describing the same rendered plot rather
   than two distinct ones.
   - *Fix-it plan*: Change cell 18 to `plt.boxplot(class_dfs[1])`, re-run
     the notebook, and verify the markdown commentary in cell 19 still
     matches the regenerated plot's actual outliers.

### Correctness Notes

6. **DBSCAN hyperparameter search's scoring silently fails on every fold**
   -- `seed_analysis_dbscan.ipynb`, cells 39-40 and 43. The grid search is
   set up as:
   ```python
   grid_search = GridSearchCV(pipeline, param_grid, cv=5,
       scoring=make_scorer(silhouette_score, greater_is_better=True))
   ```
   `silhouette_score` is an unsupervised metric with signature
   `(X, labels)`, but `make_scorer` wraps it as a supervised scorer that
   calls it as `scorer(estimator, X_test, y_true)`. On the saved run, cell
   40's output shows this raised
   `TypeError: _BaseScorer.__call__() missing 1 required positional
   argument: 'y_true'` on every train/test split, caught by scikit-learn
   as a non-fatal "Scoring failed... set to nan" warning. Consequently
   `grid_search.best_score_` is literally `nan` (printed in cell 43), and
   because every candidate's mean score is also `nan`, scikit-learn's
   `best_params_` selection is not actually choosing a best-scoring
   configuration -- it reduces to whichever candidate the tie-break lands
   on. The `eps=0.7, min_samples=4` "best params" reported afterward
   should therefore be treated as an untuned/arbitrary grid point, not a
   validated result. (Separately, both the pre-tuning silhouette in cell 31
   and the post-tuning silhouette in cell 42 are computed against
   `df_unlabeled`, the raw unscaled 7-feature space -- not the pipeline's
   PCA(3) space where DBSCAN actually formed the clusters. That makes the
   -0.214 -> 0.144 change an apples-to-apples comparison between the two
   runs, but neither number reflects the distance metric DBSCAN itself
   clustered on.)
   - *Fix-it plan*: `make_scorer` cannot wrap an unsupervised metric like
     `silhouette_score` for use with `GridSearchCV`, regardless of its
     keyword arguments. Replace it with a plain callable of the form
     `def dbscan_silhouette(estimator, X): return
     silhouette_score(X, estimator.named_steps['dbscan'].labels_)` and pass
     that directly as `scoring=dbscan_silhouette`. Also consider having
     `GridSearchCV` fall back to a fixed low score (instead of `nan`) when
     a candidate produces a single cluster or all-noise labeling, since
     silhouette score is undefined in that case; and evaluate both the
     pre- and post-tuning silhouette scores in the same feature space for
     a fair comparison. This is a live, executed code path (not dead
     code), so it directly affects the validity of the reported "best"
     DBSCAN hyperparameters.

### Security

No security findings. Neither notebook contains hardcoded credentials or
secrets, deserializes untrusted data (no `pickle.load` / `joblib.load` of
any kind -- the only `joblib` reference is the dead, commented-out `dump`
block described above), executes shell commands or subprocesses, or makes
any network calls. All data is read from the local `dataset/` folder.

## Status

Both notebooks are complete and runnable end-to-end without errors. The
K-Means notebook produces a reasonably strong recovery of the known
variety labels (silhouette ~0.41, ARI ~0.81 on the primary 4-component
pipeline, consistent across its own internal hyperparameter search). The
DBSCAN notebook produces a much weaker recovery (ARI 0.007 untuned, 0.274
after grid search) and its hyperparameter search is undermined by the
`make_scorer`/silhouette incompatibility documented above (Correctness
Notes, item 6), so its reported "best" configuration should not be relied
upon without a fix. The dead code items and the DBSCAN box-plot bug
cataloged above do not prevent either notebook from running; together with
the grid-search scoring bug, they are cleanup and correctness items for
future work, not blockers to reading or re-running the notebooks as they
stand.

## Contributions

This repository has two authors; `git log` shows a clear split by notebook:

- **Daniel Leone** (GitHub: `dagron27`) created the repository, wrote
  `README.md`, `.gitignore`, `requirements.txt`, the `.github/workflows/ci.yml`
  CI workflow, and `dataset/converter.py` plus the committed dataset files.
  He also authored the initial single-notebook scaffold, split it into the
  two current notebooks, and wrote substantially all of
  **`seed_analysis_k_means.ipynb`** (the "K-means initial" commit alone adds
  ~2,400 lines), with a later follow-up commit ("Updated K-Means"). He also
  made one conflict-resolution pass on the DBSCAN notebook early on
  ("@bob changes to DBSCAN resolved").
- **CLOR2003** (GitHub handle; commits are authored as `CLOR2003
  <93290045+CLOR2003@users.noreply.github.com>`) wrote substantially all of
  **`seed_analysis_dbscan.ipynb`** (originally committed as
  `seed_analysis.ipynb`), across a series of "Created using Colab" commits
  totaling several thousand added/changed lines. CLOR2003's commits touch
  only that one notebook -- none of the README, `.gitignore`,
  `requirements.txt`, CI workflow, or `dataset/` files were authored by
  CLOR2003.

Because of this split authorship, the LICENSE file does not grant a blanket
license to the whole repository; see LICENSE for the scope of what is and
is not covered by the MIT terms.
