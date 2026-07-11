# Developer Notes

Practical guidance for contributors working on this codebase.

---

## Getting Started

```bash
git clone https://github.com/AkshayKarthickMS/Gold-Price-Prediction.git
cd Gold-Price-Prediction

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install numpy pandas matplotlib plotly scikit-learn tensorflow dash jupyter
jupyter notebook
```

**First change to make:** fix the CSV path in Cell 2.

```python
# Before (broken on other machines)
df = pd.read_csv(r"C:\Users\USER\Desktop\jupyter\Gold Price (2013-2023).csv")

# After
df = pd.read_csv("Gold Price (2013-2023).csv")
```

There is no `requirements.txt`. Pin versions locally and commit a requirements file if you add one.

---

## Environment

| Requirement | Version (from notebook metadata) |
|-------------|----------------------------------|
| Python | 3.10 |
| Jupyter kernel | `python3` (ipykernel) |

TensorFlow compatibility varies by OS and Python version. If `import tensorflow` fails, check [TensorFlow install docs](https://www.tensorflow.org/install) for your platform.

---

## Known Issues

### 1. Scaler fit on full dataset

```python
scaler.fit(df.Price.values.reshape(-1, 1))  # includes 2022 test data
```

The scaler learns min/max from the entire series, including the held-out test year. For strict backtesting, fit only on training data:

```python
scaler.fit(df.Price[:-test_size].values.reshape(-1, 1))
```

### 2. Unused import

`train_test_split` is imported in Cell 1 but never used. Remove it to avoid confusion.

### 3. Hardcoded hyperparameters

All tuning values are inline constants:

```python
window_size = 60
epochs = 150
batch_size = 32
validation_split = 0.1
```

Extract to a config dict or `config.py` when refactoring.

### 4. Softmax on regression head

```python
x = Dense(32, activation='softmax')(x)
dnn_output = Dense(1)(x)
```

`softmax` produces a probability distribution and is uncommon before a continuous output. Consider `activation='relu'` or removing the intermediate Dense layer.

### 5. Dash blocks the kernel

`app.run_server(debug=True)` runs a blocking process inside Jupyter. Use a separate terminal or script for the dashboard during iterative training work.

### 6. No model persistence

Trained weights exist only in the kernel's `model` variable. Restarting the kernel requires retraining. Add:

```python
model.save("artifacts/gold_lstm.keras")
# keras.models.load_model("artifacts/gold_lstm.keras")
```

### 7. Double inverse transform in Dash cell

```python
y_test_true_original = scaler.inverse_transform(y_test_true.reshape(-1, 1))
```

`y_test_true` is already inverse-transformed in Cell 27. The Dash cell applies `inverse_transform` again, which produces incorrect values. Fix by using `y_test_true` and `y_test_pred` directly in the dashboard.

---

## Development Workflow

### Recommended run order

1. **Kernel → Restart & Clear Output**
2. Fix CSV path
3. **Cell → Run All**
4. Record MSE / MAPE from console output
5. Optionally capture plots for documentation

### Quick iteration (skip full retrain)

For preprocessing-only changes, stop before Cell 22. For model experiments, restart from Cell 22 after sequence tensors are built.

### Adding new features

| Change | Touch points |
|--------|-------------|
| Multivariate inputs | Cells 16–21 (sequence shape), Cell 22 (input shape) |
| Different test year | Cell 12 (`test_year` filter) |
| New metrics | Cells 25–26 |
| Save model artifacts | After Cell 23 |

---

## Testing Recommendations

No tests exist today. High-value first tests if modularized:

```python
# data_cleaner
def test_removes_volume_columns():
    ...

# sequencer
def test_window_shape():
    X, y = build_sequences(data, window_size=60)
    assert X.shape[1:] == (60, 1)

# splitter
def test_2022_holdout():
    assert test_size == (df.Date.dt.year == 2022).sum()
```

For the notebook itself, consider [`nbval`](https://nbval.readthedocs.io/) or converting to scripts and using `pytest`.

---

## Refactoring Checklist

- [ ] Create `requirements.txt` with pinned versions
- [ ] Replace hardcoded CSV path with relative path or CLI argument
- [ ] Remove unused `train_test_split` import
- [ ] Fit scaler on training data only
- [ ] Fix double inverse transform in Dash cell
- [ ] Extract `define_model()` to `src/models/lstm.py`
- [ ] Add `model.save()` after training
- [ ] Add early stopping callback
- [ ] Move Dash app to standalone `app/dashboard.py`
- [ ] Add MIT license file
- [ ] Add CI notebook smoke test

---

## Performance Tips

| Scenario | Suggestion |
|----------|------------|
| Slow training | Reduce `epochs` during development; use GPU if available |
| Memory pressure | Reduce `batch_size` or `window_size` |
| Large dataset updates | Revisit `MinMaxScaler` — fit on train only, save scaler with `joblib` |
| Repeated experiments | Save model + scaler artifacts; skip retraining when tuning only the dashboard |

---

## Git Conventions

When contributing:

- Do not commit notebook outputs with absolute local paths
- Clear cell outputs before committing (`jupyter nbconvert --clear-output`)
- Keep `Gold Price (2013-2023).csv` in the repo root (dataset is part of the project)
- Document hyperparameter changes in commit messages

---

## Related Documentation

| Document | Contents |
|----------|----------|
| [architecture.md](architecture.md) | System design and data flow |
| [project-flow.md](project-flow.md) | Cell-by-cell execution guide |
| [modules.md](modules.md) | Logical module reference |
| [api.md](api.md) | Dash dashboard interface (no REST API) |
