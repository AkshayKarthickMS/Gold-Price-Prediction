# Project Flow

Step-by-step execution guide for `Gold Price Prediction.ipynb`. The notebook contains **30 active code cells** executed sequentially.

---

## Prerequisites

1. Python 3.10 environment with dependencies installed (see [README](../README.md#setup))
2. `Gold Price (2013-2023).csv` in the working directory
3. Update the hardcoded CSV path in **Cell 2** before running:

```python
df = pd.read_csv("Gold Price (2013-2023).csv")
```

---

## Execution Phases

### Phase 1 — Setup (Cell 1)

**Purpose:** Import all libraries.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split  # unused
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense, Dropout, LSTM
```

**Produces:** No variables. Loads dependencies into the kernel.

---

### Phase 2 — Load Data (Cells 2–4)

| Cell | Action | Key output |
|------|--------|------------|
| 2 | `pd.read_csv(...)` | `df` — raw DataFrame |
| 3 | Display `df` | Visual inspection |
| 4 | `df.info()` | Schema and dtypes |

---

### Phase 3 — Preprocessing (Cells 5–10)

| Cell | Action | Effect on `df` |
|------|--------|----------------|
| 5 | Drop `Vol.`, `Change %` | Removes non-OHLC columns |
| 6 | Parse dates, sort ascending, reset index | Chronological order |
| 7 | Strip commas, cast numerics to `float64` | Clean numeric columns |
| 8 | `df.head()` | Preview cleaned data |
| 9 | `df.duplicated().sum()` | Duplicate check |
| 10 | `df.isnull().sum().sum()` | Null check |

**Checkpoint:** `df` is clean, sorted, and ready for analysis.

---

### Phase 4 — Exploratory Analysis (Cells 11–13)

| Cell | Action | Output |
|------|--------|--------|
| 11 | Plotly line chart of `Price` over `Date` | Interactive EDA figure |
| 12 | Compute `test_size` from 2022 row count | Integer — number of test rows |
| 13 | Matplotlib train/test split visualization | Black = pre-2022, blue = 2022 |

**Key variable:** `test_size` drives all subsequent slicing.

---

### Phase 5 — Feature Engineering (Cells 14–21)

| Cell | Action | Key variables |
|------|--------|---------------|
| 14 | Fit `MinMaxScaler` on full `Price` | `scaler` |
| 15 | Set `window_size = 60` | `window_size` |
| 16 | Transform training prices | `train_data` (scaled array) |
| 17 | Build training sequences | `X_train`, `y_train` (lists) |
| 18 | Slice and transform test prices (+ 60-day buffer) | `test_data` |
| 19 | Build test sequences | `X_test`, `y_test` (lists) |
| 20 | Convert lists to NumPy arrays | Typed arrays |
| 21 | Reshape to LSTM input format | `(N, 60, 1)` tensors |

**Sequence logic (training):**

```python
for i in range(window_size, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
```

**Test buffer:** `df.Price[-test_size-60:]` ensures the first 2022 sample has 60 prior days of context.

---

### Phase 6 — Model Definition (Cell 22)

**Purpose:** Define `define_model()` — a factory function returning a compiled Keras model.

**Input shape:** `(window_size, 1)` = `(60, 1)`

**Returns:** Compiled model ready for `.fit()`.

---

### Phase 7 — Training (Cell 23)

```python
model = define_model()
history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)
```

**Produces:**

- `model` — trained weights in memory
- `history` — per-epoch loss metrics (not persisted)

**Expected duration:** Several minutes depending on hardware. No GPU requirement is documented.

---

### Phase 8 — Evaluation (Cells 24–27)

| Cell | Action | Output |
|------|--------|--------|
| 24 | `model.evaluate()` and `model.predict()` | `result`, `y_pred` |
| 25 | Compute MAPE and derived accuracy | `MAPE`, `Accuracy` |
| 26 | Print metrics | Console output |
| 27 | Inverse-transform test labels and predictions | `y_test_true`, `y_test_pred` |

**Console output format:**

```
Test Loss: <float>
Test MAPE: <float>
Test Accuracy: <float>
```

---

### Phase 9 — Static Visualization (Cell 28)

Matplotlib chart with three series:

1. Training data (black) — inverse-scaled `train_data`
2. Actual 2022 prices (blue) — `y_test_true`
3. Predicted 2022 prices (red) — `y_test_pred`

---

### Phase 10 — Dash Dashboard (Cell 29)

**Purpose:** Launch a local web application.

1. Re-imports `dash`, `dcc`, `html`, `px`, `pd`, `MinMaxScaler`
2. Inverse-transforms arrays for display
3. Builds `app.layout` with two `dcc.Graph` components
4. Starts server: `app.run_server(debug=True, port=8051)`

**Access:** Open `http://127.0.0.1:8051` in a browser.

**Note:** This cell blocks the notebook kernel while the server runs. Interrupt the kernel (or stop the cell) to continue.

---

### Phase 11 — Empty Cell (Cell 30)

No code. Reserved for extensions.

---

## Flow Diagram

```
[Imports]
    │
    ▼
[Load CSV] ──► [Clean & Validate]
    │
    ▼
[EDA Plots] ──► [Compute test_size]
    │
    ▼
[Fit Scaler] ──► [Build Sequences]
    │
    ▼
[Define Model] ──► [Train 150 epochs]
    │
    ▼
[Evaluate] ──► [Inverse Transform]
    │
    ├──► [Matplotlib Plot]
    └──► [Dash Server :8051]
```

---

## Common Execution Issues

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `FileNotFoundError` on CSV load | Hardcoded path in Cell 2 | Use relative path to repo CSV |
| Shape mismatch errors | Cells run out of order | Kernel → Restart & Run All |
| Dash port in use | Previous server still running | Kill process on port 8051 or change port |
| TensorFlow import errors | Missing or incompatible TF install | Reinstall `tensorflow` for your Python version |
| Very long training | 150 epochs, no early stopping | Expected behavior; reduce `epochs` for quick tests |

---

## Re-running Subsets

| Goal | Minimum cells to re-run |
|------|-------------------------|
| Change window size | 15 → 21 → 22 → 23+ |
| Change test year | 12 → 13 → 16–29 |
| Re-train only | 22 → 23 → 24–29 |
| Refresh plots only | 28–29 (requires `model`, `scaler`, predictions) |

Always restart the kernel when changing preprocessing logic to avoid stale variables.
