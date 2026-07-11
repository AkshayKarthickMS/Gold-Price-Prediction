# Gold Price Prediction

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Last Commit](https://img.shields.io/github/last-commit/AkshayKarthickMS/Gold-Price-Prediction)

LSTM-based time-series forecaster for daily gold closing prices, trained on historical market data (2013–2023) and evaluated on a held-out 2022 test period.

## Documentation

| Document | Description |
|----------|-------------|
| [architecture.md](docs/architecture.md) | System design, data flow, and dependency graph |
| [project-flow.md](docs/project-flow.md) | Cell-by-cell notebook execution guide |
| [modules.md](docs/modules.md) | Logical module reference (notebook sections) |
| [api.md](docs/api.md) | Dash dashboard interface (no REST API) |
| [developer-notes.md](docs/developer-notes.md) | Setup, known issues, and refactoring guidance |

> **Note:** The codebase has no standalone `.py` files. All logic lives in `Gold Price Prediction.ipynb`.

---

## Problem

Forecast short-horizon gold price movement from historical daily prices without leaking future information into the evaluation set. The goal is a reproducible training pipeline, measurable forecast error, and a clear way to inspect model output—not a production trading system.

---

## Architecture

```
CSV (2013–2023)
      │
      ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ Preprocess  │───▶│ MinMax scale │───▶│ 60-day windows  │
│ sort · cast │    │ (Price)      │    │ → (N, 60, 1)    │
└─────────────┘    └──────────────┘    └────────┬────────┘
                                                  │
                     ┌────────────────────────────┘
                     ▼
              ┌──────────────┐    ┌──────────────────┐
              │ 3× LSTM-64   │───▶│ MSE · MAPE eval  │
              │ + Dropout    │    │ inverse scale    │
              └──────────────┘    └────────┬─────────┘
                                             │
                              ┌──────────────┴──────────────┐
                              ▼                             ▼
                       Matplotlib plots              Dash app (:8051)
```

| Stage | Input | Output |
|-------|-------|--------|
| Preprocessing | Raw OHLCV CSV | Chronological `float64` series; `Vol.` and `Change %` dropped |
| Split | Full series | Train: pre-2022 · Test: 2022 (calendar-year holdout) |
| Sequencing | Scaled prices | Supervised pairs: 60 prior days → next-day price |
| Inference | Test windows | Scaled predictions → inverse-transformed USD prices |

---

## Technical Decisions

**Temporal holdout over random split.** Data is split by calendar year (2022 = test). Random `train_test_split` is imported but unused. This preserves time ordering and gives a more realistic forecast evaluation than i.i.d. sampling.

**Univariate target.** Only the `Price` (close) column is modeled. `Open`, `High`, and `Low` remain in the dataset but are not used as features—a deliberate simplification for a single-series baseline.

**60-day lookback window.** Each sample uses the prior 60 trading days to predict the next price. This balances context length against sequence count and memory footprint for the LSTM input tensor.

**Stacked LSTM with dropout.** Three LSTM layers (64 units) with 0.2 dropout between layers. Dropout acts as regularization on a relatively small univariate series to limit overfitting during 150-epoch training.

**Nadam + MSE.** The optimizer and loss function are standard choices for continuous regression. Validation loss is monitored via a 10% holdout from the training sequences.

**MAPE for interpretability.** Test error is reported as MSE (model loss) and MAPE (business-readable percentage error). A derived `1 - MAPE` metric is also printed for quick comparison.

**Dual visualization layer.** Matplotlib handles static train/test/prediction plots. Dash serves the same results interactively for local inspection without re-running analysis cells.

---

## Performance Considerations

| Area | Current approach | Trade-off |
|------|------------------|-----------|
| Training cost | 150 epochs, batch size 32 | Thorough convergence search; no early stopping, so runtime is fixed per run |
| Input dimensionality | Univariate `(60, 1)` tensors | Lower memory and faster training vs. multivariate OHLC inputs |
| Scaling | `MinMaxScaler` fit on full `Price` column before split | Simplifies implementation; test-set statistics leak into the scaler—acceptable for a prototype, not for rigorous backtesting |
| Test window overlap | Test sequences include 60 days before 2022 | Required for the first test prediction; standard for sliding-window evaluation |
| Model persistence | Weights not saved to disk | Each run retrains from scratch; no cold-start inference path |

> The notebook does not commit execution outputs. Run locally to obtain concrete MSE and MAPE values.

---

## Code Organization

```
Gold-Price-Prediction/
├── docs/
│   ├── architecture.md
│   ├── project-flow.md
│   ├── modules.md
│   ├── api.md
│   └── developer-notes.md
├── Gold Price (2013-2023).csv   # Source data (~2,500 daily records)
├── Gold Price Prediction.ipynb  # Full pipeline: EDA → train → evaluate → Dash
└── README.md
```

The project is a **single Jupyter notebook**—appropriate for exploration and portfolio demonstration, but not yet structured for team development or deployment.

| Concern | Status |
|---------|--------|
| Modularity | All logic in one notebook; no `src/` package or config layer |
| Reproducibility | No `requirements.txt` or pinned versions |
| Portability | CSV path is hardcoded to a local Windows directory |
| Configuration | Hyperparameters (`window_size=60`, `epochs=150`) are inline constants |
| Dead code | `train_test_split` imported but unused |

---

## Scalability & Maintainability

**What scales today**
- The pipeline pattern (preprocess → sequence → train → evaluate) transfers directly to other univariate financial series.
- Dash decouples result inspection from the training loop once predictions are materialized.

**Current limits**
- Retraining on new data requires manual notebook edits and a full 150-epoch run.
- No model artifact, logging, or experiment tracking (MLflow, W&B, etc.).
- No automated tests or CI to catch regressions in preprocessing or tensor shapes.
- Dash runs as an embedded notebook cell (`debug=True`); not packaged for multi-user or cloud deployment.

**Logical next steps** (not yet implemented)
- Extract modules: `data.py`, `model.py`, `train.py`, `app.py`
- Add `requirements.txt`, relative paths, and `model.save()` / `load_model()`
- Fit scaler on training data only; add early stopping
- Extend to multivariate inputs (`Open`, `High`, `Low`) behind a feature flag

---

## Tech Stack

Python 3.10 · Pandas · NumPy · scikit-learn · TensorFlow/Keras · Matplotlib · Plotly · Dash · Jupyter

---

## Setup

```bash
git clone https://github.com/AkshayKarthickMS/Gold-Price-Prediction.git
cd Gold-Price-Prediction

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install numpy pandas matplotlib plotly scikit-learn tensorflow dash jupyter
jupyter notebook
```

Update the dataset path before running:

```python
df = pd.read_csv("Gold Price (2013-2023).csv")
```

Run all cells in order. The Dash server starts at `http://127.0.0.1:8051` after the final cell.

---

## Evaluation Output

```
Test Loss: <mse>
Test MAPE: <mape>
Test Accuracy: <1 - mape>
```

Plots compare training history, 2022 actual prices, and model predictions on the held-out period.
