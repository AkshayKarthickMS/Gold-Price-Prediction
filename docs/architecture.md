# Architecture

This document describes the system design of the Gold Price Prediction project as implemented today.

## Scope

The project is a **single-notebook ML pipeline**. There are no standalone `.py` modules, services, or deployment artifacts. All logic lives in `Gold Price Prediction.ipynb`.

| Component | Role |
|-----------|------|
| `Gold Price (2013-2023).csv` | Static input dataset |
| `Gold Price Prediction.ipynb` | Data prep, model training, evaluation, visualization |
| Dash (embedded in notebook) | Local web UI for inspecting results |

There is no REST API, database, job scheduler, or model registry.

---

## High-Level Design

```
┌──────────────────────────────────────────────────────────────────┐
│                     Gold Price Prediction.ipynb                  │
├──────────────┬──────────────┬──────────────┬─────────────────────┤
│  Data Layer  │  ML Layer    │  Eval Layer  │  Presentation Layer │
│              │              │              │                     │
│  CSV load    │  Sequences   │  MSE / MAPE  │  Matplotlib plots   │
│  Clean/sort  │  LSTM model  │  Inverse     │  Dash dashboard     │
│  EDA         │  Training    │  transform   │  (localhost:8051)   │
└──────────────┴──────────────┴──────────────┴─────────────────────┘
         │              │              │                  │
         ▼              ▼              ▼                  ▼
    Pandas/NumPy   TensorFlow/Keras  scikit-learn    Matplotlib/Plotly/Dash
```

The pipeline is **linear and stateful**: each notebook cell depends on variables created in earlier cells (`df`, `scaler`, `model`, `y_pred`, etc.). There is no dependency injection or configuration layer.

---

## Data Flow

### 1. Ingestion

- Source: `Gold Price (2013-2023).csv`
- Loader: `pd.read_csv()` (currently hardcoded to an absolute Windows path)
- Raw columns: `Date`, `Price`, `Open`, `High`, `Low`, `Vol.`, `Change %`

### 2. Preprocessing

| Step | Operation | Result |
|------|-----------|--------|
| Column filter | Drop `Vol.`, `Change %` | OHLC + `Price` retained |
| Date parsing | `pd.to_datetime` | Typed datetime column |
| Ordering | Sort ascending by `Date` | Chronological series |
| Type coercion | Strip commas, cast to `float64` | Numeric OHLC fields |
| QA | `duplicated()`, `isnull()` checks | Manual inspection only |

Only `Price` is used downstream. `Open`, `High`, and `Low` are cleaned but not fed into the model.

### 3. Train / Test Partition

Split is **time-based**, not random:

```python
test_size = df[df.Date.dt.year == 2022].shape[0]
```

- **Training**: all rows before the 2022 block
- **Test**: all rows where `Date.year == 2022`

This avoids shuffling future data into the training set.

### 4. Scaling

```python
scaler = MinMaxScaler()
scaler.fit(df.Price.values.reshape(-1, 1))
```

`MinMaxScaler` maps prices to `[0, 1]`. The scaler is fit on the **entire** `Price` column (including 2022), then applied separately to train and test slices. See [developer-notes.md](developer-notes.md) for implications.

### 5. Sequence Construction

Hyperparameter: `window_size = 60`

For each index `i >= 60`:

- **Input** `X`: scaled prices at indices `[i-60, i)`
- **Target** `y`: scaled price at index `i`

Tensors are reshaped to:

| Array | Shape |
|-------|-------|
| `X_train`, `X_test` | `(samples, 60, 1)` |
| `y_train`, `y_test` | `(samples, 1)` |

Test sequences include 60 days of context immediately before 2022 so the first 2022 prediction has sufficient history.

### 6. Model

`define_model()` builds a Keras functional model:

```
Input(60, 1)
  → LSTM(64, return_sequences=True) → Dropout(0.2)
  → LSTM(64, return_sequences=True) → Dropout(0.2)
  → LSTM(64)                        → Dropout(0.2)
  → Dense(32, activation='softmax')
  → Dense(1)
```

Compiled with `loss='mean_squared_error'` and `optimizer='Nadam'`.

### 7. Training & Evaluation

```python
model.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.1)
result = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
```

Metrics printed:

- `Test Loss` — MSE on scaled test targets
- `Test MAPE` — `mean_absolute_percentage_error(y_test, y_pred)`
- `Test Accuracy` — `1 - MAPE`

Predictions are inverse-transformed with `scaler.inverse_transform()` before plotting.

### 8. Presentation

Two output channels share the same underlying arrays:

1. **Matplotlib** — static overlay of training data, actual test prices, and predictions
2. **Dash** — interactive Plotly charts served at `http://127.0.0.1:8051`

---

## Dependency Graph

```
df
 ├── test_size
 ├── scaler ──► train_data, test_data
 │                ├── X_train, y_train
 │                └── X_test, y_test
 │                      └── model ──► y_pred
 │                             ├── MAPE, result
 │                             └── y_test_true, y_test_pred
 └── df.Date (used in all plots)
```

Re-running cells out of order will produce inconsistent state. Always execute top-to-bottom after a kernel restart.

---

## External Dependencies

| Library | Usage |
|---------|-------|
| `pandas` | CSV I/O, datetime handling, filtering |
| `numpy` | Array construction and reshaping |
| `sklearn.preprocessing.MinMaxScaler` | Price normalization |
| `sklearn.metrics.mean_absolute_percentage_error` | Test metric |
| `tensorflow` / `keras` | LSTM model definition and training |
| `matplotlib` | Static charts |
| `plotly.express` | EDA line chart and Dash figures |
| `dash` | Local dashboard server |

`train_test_split` is imported but never called.

---

## What This Architecture Does Not Include

- Model serialization (`model.save()`)
- Configuration files or environment variables
- Automated retraining or data refresh
- Authentication or multi-tenant serving
- Unit / integration tests
- CI/CD pipeline

See [developer-notes.md](developer-notes.md) for recommended evolution paths.
