# Basis-expansion parameterization of 2D photoluminescence maps

[Русская версия](README_ru.md)

This repository studies whether two-dimensional photoluminescence (PL)
excitation-emission maps can be represented by a small set of expansion
coefficients without sacrificing the accuracy of the inverse problem. The
target task is to predict the concentrations of seven ions from each PL map:
Cu²⁺, Ni²⁺, Pb²⁺, Al³⁺, Co²⁺, Cr³⁺, and NO₃⁻.

The central idea is to replace a `27 × 201` map (5,427 intensity values) with
approximately 100 coefficients of a fixed functional basis. This controlled
dimensionality reduction makes it possible to use compact classical regression
models on a spectroscopy dataset of 7,813 samples and compare them with a
convolutional neural-network baseline.

## Scientific approach

For a map $F(x,y)$, each transformer evaluates projections or moments of the
form

$$
c_k \approx \sum_{i,j} F(x_i,y_j)\,\phi_k(x_i,y_j)\,\Delta x\,\Delta y,
$$

where $\phi_k$ belongs to one of four basis families. Increasing the maximum
order increases the number of retained coefficients.

- **Polynomial moments** — central power moments arranged by total order. They
  give an interpretable description of distribution shape but do not form an
  orthogonal basis and may become unstable at high orders. See
  [MomentumTransformer2D](descriptions/MomentumTransformer2D.md).
- **Legendre expansion** — projections onto products of Legendre polynomials on
  the normalized rectangle $[-1,1]^2$. Orthogonality tends to reduce
  redundancy between features. See
  [LegendreTransformer2D](descriptions/LegendreTransformer2D.md).
- **Fourier expansion** — sine/cosine projections on the normalized rectangle,
  containing the valid cosine-cosine, cosine-sine, sine-cosine, and sine-sine
  modes. See [FourierTransformer2D](descriptions/FourierTransformer2D.md).
- **Zernike expansion** — radial and angular Zernike modes on a unit disk. In
  the current experiment the physical bounds are `x=(375, 575)` nm and
  `y=(280, 410)` nm, the fixed center is `(441, 350)` in `(x, y)` coordinates,
  and the radius reaches the most distant rectangle corner
  (`radius_mode="max"`). See
  [ZernikeTransformer2D](descriptions/ZernikeTransformer2D.md).

All implementations follow the scikit-learn transformer interface and are
collected in [src.py](src.py). A concise API-oriented overview is available in
the [usage guide](descriptions/Usage_Guide.md), while
[Momentum_transform.ipynb](Momentum_transform.ipynb) demonstrates coefficient
calculation and reconstruction on synthetic signals.

## Dataset and preprocessing

The dataset is not distributed with this repository. Place it at the repository
root using the following layout:

```text
CD_HM_dataset/
├── Y_ions.csv
├── 1.csv
├── 2.csv
├── ...
└── 7813.csv
```

`Y_ions.csv` must use the sample number as the `num` index and contain the
columns `Cu`, `Ni`, `Pb`, `Al`, `Co`, `Cr`, and `NO3`. Each numbered CSV stores
one `27 × 201` PL map. The rows correspond to excitation wavelengths from 280
to 410 nm and the columns to emission wavelengths from 375 to 575 nm.

During loading, [import_df](src.py) converts negative intensities to missing
values; the experiment replaces them with zero. All maps are divided by one
global maximum computed over the complete dataset. For each ion and CV fold, a
`MinMaxScaler` is fitted only to the training targets, and predictions are
converted back to the original concentration scale before metrics are
calculated.

## Experiment design

The main experiment is implemented in
[Experiment_1.ipynb](Experiment_1.ipynb):

- shuffled 10-fold cross-validation with `random_state=42`;
- one independent regression problem per ion;
- orders `0...13` for polynomial, Legendre, and Zernike features;
- orders `0...7` for Fourier features;
- 1–105 features for the first three order grids and 1–113 Fourier features;
- linear regression (LR), random forest (RF), XGBoost gradient boosting (GB),
  and a PyTorch multilayer perceptron (MLP);
- a PyTorch CNN baseline trained directly on the `27 × 201` maps;
- MAE, RMSE, and $R^2$ recorded independently for every fold.

The early-stopping models use an internal validation fraction of `2/7`. The
current CNN contains two convolutional layers and a 32-unit hidden linear layer.
Basis features are cached by basis, order, and fold to reduce repeated work.

The primary result structure is

```text
results[model][basis][ion][n_features][metric] = [10 fold values]
```

and the CNN baseline uses

```text
baseline[ion][metric] = [10 fold values]
```

## Results

The archived study outputs are stored in
[results/Experiment_1_3.json](results/Experiment_1_3.json) and
[results/Baseline_1_3.json](results/Baseline_1_3.json). The former contains the
complete LR, RF, and GB sweeps. Its MLP section is partial and contains only the
polynomial-moment orders 0–4; it should not be interpreted as a complete
four-basis comparison.

The archived experiment supports the following cautious conclusions:

- roughly `80 ± 5` coefficients provide a useful compromise between information
  retention and model complexity, although the optimum depends on the ion;
- compact classical models can achieve errors comparable with the CNN baseline;
- Legendre and Fourier features are generally among the most stable because
  they are orthogonal and cover the full rectangular map;
- gradient boosting gives the most consistently strong performance among the
  evaluated classical models;
- very high polynomial orders can increase both the mean error and its
  cross-validation variance.

These observations describe this dataset and protocol; they are not claims of
universal superiority of one basis or model.

### Version note

The archived JSON files used for the reported analysis were generated before
the current fixed-center/max-radius Zernike configuration and the current
two-convolution CNN correction were introduced. Consequently, rerunning the
current [Experiment_1.ipynb](Experiment_1.ipynb) may produce different Zernike
and CNN results. The supplied JSON files should be treated as the historical
study snapshot, while a fresh run evaluates the current implementation.

## Reproduction

### 1. Environment

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/Gavr101/Spectroscopy_FL_7_momenta.git
cd Spectroscopy_FL_7_momenta
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows
PowerShell or `source .venv/bin/activate` on Linux/macOS. The notebooks can be
opened directly in VS Code. To use JupyterLab instead, install it separately:

```bash
python -m pip install jupyterlab
python -m jupyter lab
```

CUDA is optional; PyTorch falls back to CPU when no compatible GPU is available.

### 2. Recreate the figures without training

Open [Image_generation.ipynb](Image_generation.ipynb) for English figures or
[Генерация изображений.ipynb](Генерация%20изображений.ipynb) for Russian
figures. Both read the archived JSON files from `results/`, so the external
dataset and model retraining are not required.

The notebooks produce:

- MAE and $R^2$ curves versus feature count for LR, RF, and GB;
- four basis-wise bar charts comparing CNN, LR, RF, and GB near 80 features;
- three model-wise bar charts comparing the four basis families.

### 3. Rerun the complete experiment

1. Place `CD_HM_dataset/` at the repository root.
2. Open [Experiment_1.ipynb](Experiment_1.ipynb) from that root.
3. To preserve the archived study, change `RESULTS_PATH`, `BASELINE_PATH`, and
   `CACHE_DIR` to new names before running. Reusing an old cache would mix
   features from different transformer configurations.
4. Run all cells. The notebook saves results after every fold and can resume an
   interrupted run.
5. Point the visualization notebook to the newly generated JSON files.

The full sweep is computationally expensive, especially for MLP and CNN models.

## Repository map

- [src.py](src.py) — reusable 1D and 2D transformers and the PL-map CSV loader.
- [Momentum_transform.ipynb](Momentum_transform.ipynb) — mathematical and
  reconstruction demonstrations.
- [Experiment_1.ipynb](Experiment_1.ipynb) — preprocessing, cross-validation,
  model training, caching, and result serialization.
- [descriptions/](descriptions/) — mathematical descriptions and usage notes.
- [results/](results/) — archived fold-level metrics and generated figures.
- [Image_generation.ipynb](Image_generation.ipynb) — English result figures.
- [Генерация изображений.ipynb](Генерация%20изображений.ipynb) — Russian result
  figures.

## Reproducibility limitations

- The source dataset must be obtained separately.
- The global PL-intensity normalization uses information from all samples,
  although the target scaler is fitted within each training fold.
- The archived MLP sweep in `Experiment_1_3.json` is incomplete.
- Archived results and the current Zernike/CNN implementation correspond to
  different code states, as described in the version note.
