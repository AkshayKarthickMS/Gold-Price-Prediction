# Modules

## Python Files

There are **no standalone `.py` files** in this repository. All executable code is contained in:

```
Gold Price Prediction.ipynb
```

This document maps the notebook's **logical modules**—the functional units a future refactor would extract into separate files.

---

## Logical Module Map

| Logical module | Notebook cells | Responsibility |
|----------------|----------------|----------------|
| `imports` | 1 | Dependency loading |
| `data_loader` | 2 | CSV ingestion |
| `data_cleaner` | 5–7 | Column filtering, date parsing, type coercion |
| `data_validator` | 8–10 | Preview, duplicate, and null checks |
| `eda` | 3–4, 11, 13 | Exploratory displays and charts |
| `splitter` | 12 | Compute temporal test set size |
| `scaler` | 14, 16, 18 | Fit and apply `MinMaxScaler` |
| `sequencer` | 15, 17, 19–21 | Build sliding-window tensors |
| `model` | 22 | `define_model()` factory |
| `trainer` | 23 | Model fitting loop |
| `evaluator` | 24–27 | Loss, MAPE, inverse transform |
| `plotter` | 28 | Matplotlib performance chart |
| `dashboard` | 29 | Dash application |

---

## Module Details

### `imports` (Cell 1)

**Exports:** None (side-effect imports only)

| Import | Used by |
|--------|---------|
| `numpy` | `sequencer`, `evaluator` |
| `pandas` | `data_loader`, `data_cleaner`, `eda`, `dashboard` |
| `matplotlib.pyplot` | `eda`, `plotter` |
| `plotly.express` | `eda`, `dashboard` |
| `MinMaxScaler` | `scaler`, `dashboard` |
| `mean_absolute_percentage_error` | `evaluator` |
| `tensorflow`, `keras` | `model`, `trainer`, `evaluator` |
| `train_test_split` | **Unused** — safe to remove |

---

### `data_loader` (Cell 2)

**Function equivalent:**

```python
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
```

**Current issue:** Path is hardcoded:

```python
r"C:\Users\USER\Desktop\jupyter\Gold Price (2013-2023).csv"
```

**Output:** `df`

---

### `data_cleaner` (Cells 5–7)

**Operations:**

```python
df.drop(['Vol.', 'Change %'], axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by='Date', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)

NumCols = df.columns.drop(['Date'])
df[NumCols] = df[NumCols].replace({',': ''}, regex=True)
df[NumCols] = df[NumCols].astype('float64')
```

**Mutates:** `df` in place

**Columns after cleaning:** `Date`, `Price`, `Open`, `High`, `Low`

---

### `data_validator` (Cells 8–10)

Read-only inspection. No transformations.

```python
df.head()
df.duplicated().sum()
df.isnull().sum().sum()
```

---

### `eda` (Cells 3–4, 11, 13)

Generates visual outputs only. Does not modify model inputs.

| Cell | Chart |
|------|-------|
| 11 | Plotly — full price history |
| 13 | Matplotlib — train (black) vs test (blue) split |

---

### `splitter` (Cell 12)

**Function equivalent:**

```python
def compute_test_size(df: pd.DataFrame, test_year: int = 2022) -> int:
    return df[df.Date.dt.year == test_year].shape[0]
```

**Output:** `test_size: int`

Used by all subsequent slicing: `df.Price[:-test_size]`, `df.Price[-test_size:]`, etc.

---

### `scaler` (Cells 14, 16, 18)

**Fit (Cell 14):**

```python
scaler = MinMaxScaler()
scaler.fit(df.Price.values.reshape(-1, 1))
```

**Transform training (Cell 16):**

```python
train_data = scaler.transform(df.Price[:-test_size].values.reshape(-1, 1))
```

**Transform test with buffer (Cell 18):**

```python
test_data = scaler.transform(df.Price[-test_size - 60:].values.reshape(-1, 1))
```

**Outputs:** `scaler`, `train_data`, `test_data`

---

### `sequencer` (Cells 15, 17, 19–21)

**Constants:**

```python
window_size = 60
```

**Sequence builder pattern:**

```python
def build_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
```

**Reshape for LSTM:**

```python
X = X.reshape(X.shape[0], X.shape[1], 1)  # (samples, 60, 1)
y = y.reshape(-1, 1)                       # (samples, 1)
```

**Outputs:** `X_train`, `y_train`, `X_test`, `y_test`

---

### `model` (Cell 22)

**Public interface:**

```python
def define_model() -> keras.Model:
    ...
    return model
```

**Architecture summary:**

| Layer | Config |
|-------|--------|
| Input | `shape=(60, 1)` |
| LSTM × 3 | 64 units; first two return sequences |
| Dropout × 3 | rate = 0.2 |
| Dense | 32 units, `activation='softmax'` |
| Dense (output) | 1 unit, linear (default) |

**Compilation:**

```python
model.compile(loss='mean_squared_error', optimizer='Nadam')
```

**Note:** `softmax` on a Dense layer before a regression output is atypical. A future refactor should evaluate `relu` or remove the intermediate Dense layer.

---

### `trainer` (Cell 23)

```python
model = define_model()
history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
)
```

**Hyperparameters (inline constants):**

| Parameter | Value |
|-----------|-------|
| `epochs` | 150 |
| `batch_size` | 32 |
| `validation_split` | 0.1 |

**Outputs:** `model` (trained), `history`

---

### `evaluator` (Cells 24–27)

```python
result = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
MAPE = mean_absolute_percentage_error(y_test, y_pred)
Accuracy = 1 - MAPE
y_test_true = scaler.inverse_transform(y_test)
y_test_pred = scaler.inverse_transform(y_pred)
```

| Output | Description |
|--------|-------------|
| `result` | Scalar MSE loss |
| `y_pred` | Scaled predictions |
| `MAPE` | Mean absolute percentage error |
| `Accuracy` | `1 - MAPE` (not classification accuracy) |
| `y_test_true`, `y_test_pred` | Original USD price scale |

---

### `plotter` (Cell 28)

Static Matplotlib figure. Reads `df`, `test_size`, `train_data`, `scaler`, `y_test_true`, `y_test_pred`.

No function encapsulation — plotting code is inline.

---

### `dashboard` (Cell 29)

Self-contained Dash app defined inline.

**Re-declares imports:** `dash`, `dcc`, `html`, `px`, `pd`, `MinMaxScaler`

**Entry point:**

```python
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
```

In Jupyter, `__name__` is `"__main__"`, so the server starts when the cell executes.

---

## Suggested Refactor Structure

If this notebook is modularized, the following layout is a natural target:

```
src/
├── config.py          # window_size, epochs, test_year, csv_path
├── data/
│   ├── loader.py      # data_loader
│   ├── cleaner.py     # data_cleaner
│   └── sequences.py   # scaler + sequencer
├── models/
│   └── lstm.py        # define_model()
├── train.py           # trainer
├── evaluate.py        # evaluator
├── visualize.py       # plotter
└── app/
    └── dashboard.py   # Dash app
```

None of these files exist yet. This is a reference for contributors only.

---

## Dataset Module (Non-Code)

`Gold Price (2013-2023).csv` is a static data asset, not a Python module.

| Column | Type (after cleaning) | Used in model |
|--------|----------------------|---------------|
| `Date` | `datetime64` | Plotting and splitting only |
| `Price` | `float64` | **Target and sole feature** |
| `Open` | `float64` | Cleaned, unused |
| `High` | `float64` | Cleaned, unused |
| `Low` | `float64` | Cleaned, unused |
