# Hybrid Quantitative

A compact single-file pipeline that builds a hybrid price forecast by combining GARCH volatility estimates with an LSTM-based sequence model.

The approach at a glance:
- Estimate conditional daily volatility with a GARCH(1,1) model fitted on returns.
- Compute technical features (Close price, volatility, RSI), scale them, and train a small LSTM on sliding windows.
- Run multiple noisy LSTM simulations and aggregate them to produce a robust short-term forecast.

All results (JSON + optional plots) are saved to the `results/` folder.

## Quick start

Requirements

- Python 3.10+ (development was done on Python 3.10+)
- Install pinned dependencies:

```bash
pip install -r requirements.txt
```

Run a short debug session (fast, low-cost):

```bash
python hybrid-quantitative.py --ticker VALE3.SA --start 2020-01-01 --epochs 2 --simulations 5
```

This downloads data from Yahoo Finance, estimates volatility with `arch`, trains the LSTM, runs simulations, and writes results to `results/`.

## What this repo contains

- `hybrid-quantitative.py` — single-file CLI pipeline (download -> features -> GARCH -> LSTM -> simulations -> plotting + JSON output).
- `requirements.txt` — pinned Python dependencies used for development (TensorFlow, arch, yfinance, etc.).
- `results/` — output JSON and optional saved plots.

## Features

- Historical OHLCV download using `yfinance`.
- Conditional volatility estimation using `arch` (GARCH(1,1)) on returns.
- RSI added as an extra feature; features are stacked in the order [Close, volatility, rsi].
- LSTM trained on sliding windows of recent days and used to produce multiple stochastic simulations for robust forecasting.

## CLI (common flags)

The main script supports a compact set of flags for common workflows. Typical flags you can pass on the command line:

- `--ticker` : ticker symbol used by Yahoo Finance (e.g. `VALE3.SA`).
- `--start` / `--end` : date range for historical data (ISO format YYYY-MM-DD). If `--end` is omitted the script uses today's date.
- `--epochs` : number of training epochs (use small numbers like 1–5 for quick debugging).
- `--simulations` : number of noisy LSTM simulations to run and aggregate.

Open `hybrid-quantitative.py` to see the full set of available options and defaults.

## Output

- JSON results are written to `results/` (timestamped filenames).
- Plots (if generated) are saved to `results/` alongside numeric output.

## Plot legend

- Actual price: blue line
- Hybrid prediction (LSTM + volatility + RSI): red line
- Tomorrow's forecast: dashed line (green if predicted up, red if predicted down) with an annotated price and arrow indicator.

## Implementation notes & gotchas

- GARCH must be fit on daily returns (not raw prices) to yield meaningful volatility estimates.
- After adding derived columns (`returns`, `volatility`, `rsi`) the code calls `data.dropna()` to keep rows aligned.
- `MinMaxScaler` is fit on the full feature matrix `[Close, volatility, rsi]`. When inverse-transforming a single predicted Close value the script pads the remaining columns with zeros (same column order) before calling `scaler.inverse_transform` to avoid a shape mismatch.
- TensorFlow prefers `float32` inputs; arrays are cast before training. The script uses optimizer clipping and `TerminateOnNaN` to protect training.
- GARCH fitting (`arch`) can be slow on long histories — for development use shorter date ranges or cache the fitted volatility series.

## Development tips

- For fast iteration: use `--epochs 1` and a small `--simulations` value to verify data flow and shapes.
- Inspect helper functions inside `hybrid-quantitative.py`: `download_data(...)`, `calculate_volatility(...)`, `prepare_lstm_data(...)`, `train_lstm(...)`, and `predict_tomorrow(...)` for fine-grained control.

## Example quick debug

Run a minimal end-to-end test that completes quickly:

```bash
python hybrid-quantitative.py --ticker VALE3.SA --start 2020-01-01 --epochs 1 --simulations 3
```

This prints debug shapes, a few sample predictions, and writes a JSON file under `results/`.

## License & disclaimer

This project is provided for research and educational use only. The forecasts are illustrative and do not constitute financial advice.
