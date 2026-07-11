# API Reference

## REST API

**Not applicable.** This project does not expose HTTP endpoints, webhooks, or a backend service. There is no `Flask`, `FastAPI`, or similar API layer.

---

## Local Dashboard Interface

The only network-facing component is the **Dash development server** embedded in the final notebook cell. It serves a read-only visualization UI—not a programmatic API.

### Server

| Property | Value |
|----------|-------|
| Framework | [Dash](https://dash.plotly.com/) |
| Host | `127.0.0.1` (default) |
| Port | `8051` |
| Debug mode | `True` |
| Entry point | `app.run_server(debug=True, port=8051)` |

### Starting the Server

Run all notebook cells through the Dash cell, then open:

```
http://127.0.0.1:8051
```

The server runs inside the Jupyter kernel process. Stopping the cell or restarting the kernel terminates the server.

---

## Dashboard Layout

```python
app.layout = html.Div([
    html.H1("Gold Price Prediction Model Performance"),
    dcc.Graph(figure=<training_data_plot>),
    dcc.Graph(figure=<actual_vs_predicted_plot>),
])
```

### Component 1 — Page Title

| Element | Type | Content |
|---------|------|---------|
| Heading | `html.H1` | `"Gold Price Prediction Model Performance"` |

### Component 2 — Training Data Chart

| Property | Value |
|----------|-------|
| Component | `dcc.Graph` |
| Chart type | Plotly line (`px.line`) |
| X-axis | `df['Date'].iloc[:-test_size]` |
| Y-axis | Inverse-scaled training prices |
| Template | `plotly_dark` |
| Title | `"Training Data"` |

### Component 3 — Test Comparison Chart

| Property | Value |
|----------|-------|
| Component | `dcc.Graph` |
| Chart type | Plotly line (`px.line`) |
| X-axis | `df['Date'].iloc[-test_size:]` |
| Y-axis | Two series — actual and predicted 2022 prices |
| Template | `plotly_dark` |
| Title | `"Actual vs. Predicted Test Data"` |

---

## Data Contract (In-Memory)

The dashboard reads precomputed notebook variables. It does not accept user input or query parameters.

| Variable | Type | Description |
|----------|------|-------------|
| `df` | `pd.DataFrame` | Full cleaned dataset with `Date` and `Price` |
| `test_size` | `int` | Number of 2022 test rows |
| `train_data` | `np.ndarray` | Scaled training prices |
| `y_test_true` | `np.ndarray` | Actual test prices (original scale) |
| `y_test_pred` | `np.ndarray` | Predicted test prices (original scale) |
| `scaler` | `MinMaxScaler` | Fitted scaler for inverse transforms |

If any of these are undefined (cells not yet run), the Dash cell will raise a `NameError`.

---

## Interaction Model

| Feature | Supported |
|---------|-----------|
| Zoom / pan on charts | Yes (Plotly default) |
| Hover tooltips | Yes |
| User-supplied date range | No |
| Model re-prediction on demand | No |
| Authentication | No |
| JSON / CSV export | No |

---

## Adding an API (Future)

If the project is refactored into a service, a minimal inference API might look like:

```
POST /predict
  Body: { "history": [float × 60] }
  Response: { "predicted_price": float }
```

This does not exist today. See [developer-notes.md](developer-notes.md) for refactoring guidance.
