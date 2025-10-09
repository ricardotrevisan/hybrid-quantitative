# hybrid_predict_full_plot.py
import argparse
import yfinance as yf
import pandas as pd
from pandas import Timestamp
import numpy as np
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import json
from datetime import timedelta
import seaborn as sns



# -----------------------------
# Feature Selection
# -----------------------------
def remove_highly_correlated_features(df, threshold=0.95, verbose=True):
    """Remove features with correlation above the specified threshold."""
    corr_matrix = df.corr().abs()  
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )  # Triângulo superior da matriz
    
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    
    if verbose and to_drop:
        print("Features removidas por alta correlação:", to_drop)
    
    return df.drop(columns=to_drop)


# -----------------------------
# JSON Utils
# -----------------------------
def convert_json_compatible(obj):
    """Converts non-serializable objects to native Python types."""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, Timestamp):
        return obj.isoformat()
    else:
        return obj

def save_results_to_json(results, filename="results.json"):
    """Save results list to a JSON file."""
    # Convert all elements recursively to JSON-safe formats
    clean_results = [
        {k: convert_json_compatible(v) for k, v in item.items()} for item in results
    ]
    with open(filename, "a", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=4, ensure_ascii=False)
    print(f"✅ Results saved to {filename}")


# -----------------------------
# Download and preparation
# -----------------------------
def download_data(ticker='VALE3.SA', start="2020-01-01", end=None):
    """ Download stock data and compute returns. """
    raw_data = yf.download(ticker, start=start, end=None, auto_adjust=True)
    data = raw_data[['Close', 'Volume']].copy()

    # compute returns and drop rows with NaNs in the DataFrame once
    #data['returns'] = data['Close'].pct_change()
    data['returns'] = np.log(data['Close'] / data['Close'].shift(1)) #melhor para o que pretendemos

    data.dropna(inplace=True)

    last_day = data.iloc[-1]
    close = float(last_day['Close'].iloc[0]) if isinstance(last_day['Close'], pd.Series) else float(last_day['Close'])
    volume = float(last_day['Volume'].iloc[0]) if isinstance(last_day['Volume'], pd.Series) else float(last_day['Volume'])
    returns = float(last_day['returns'])

    print(f"\nLast day available: {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Last Close: {close:.2f}, Volume: {volume:.0f}, Return: {returns:.4f}\n")

    if end is not None:
        data = data.loc[data.index <= pd.to_datetime(end)]

    return data


# -----------------------------
# VIX
# -----------------------------
def get_vix_returns(data, start="2020-01-01", end=None, interval='1d', log_returns=True, fill_method='ffill'):
    """ Adiciona retornos do VIX ao DataFrame principal.  """
    # Garantir índice datetime no DataFrame principal
    data = data.copy()
    data.index = pd.to_datetime(data.index)
    
    # Baixar dados do VIX
    vix = yf.download('^VIX', start=start, end=end, interval=interval, progress=False)
    if vix.empty:
        raise ValueError("Não foi possível baixar dados do VIX.")

    # Calcular retornos
    if log_returns:
        vix['VIX_Returns'] = np.log(vix['Close'] / vix['Close'].shift(1))
    else:
        vix['VIX_Returns'] = vix['Close'].pct_change()

    vix_returns = vix[['VIX_Returns']].dropna()
    vix_returns.index = pd.to_datetime(vix_returns.index)
    
    # Alinhar com o DataFrame principal
    # Resample/reindex para o índice do ativo e preencher valores faltantes
    vix_aligned = vix_returns.reindex(data.index, method=fill_method)
    
    # Adicionar ao DataFrame original
    data['VIX_Returns'] = vix_aligned['VIX_Returns']
    
    return data


# -----------------------------
# weight decay
# -----------------------------
def add_weight_decay(data, weight=0.01):
    """Adiciona um fator de decaimento de peso ao DataFrame."""
    # lower weight = slower decay
    # should consider special events (e.g., COVID crash) - addressed by rolling z-score, VIX & walking forward
    decay = weight
    df = data.copy()
    df['sw'] = np.exp(-decay * (df.index.max() - df.index).days)
    return df


# -----------------------------
# z-score returns
# -----------------------------
def add_rolling_z_score(data, windows=[30, 45, 60, 90]):
    for window in windows:
        data[f'z{window}'] = (data['returns'] - data['returns'].rolling(window).mean()) / data['returns'].rolling(window).std()
    data.dropna(inplace=True)
    return data


# -----------------------------
# GARCH Volatility estimation
# -----------------------------
def calculate_volatility(data):
    garch = arch_model(data['returns'] * 100, vol='Garch', p=1, q=1)
    garch_fit = garch.fit(disp="off")
    data['volatility'] = garch_fit.conditional_volatility
    data.dropna(inplace=True)
    return data


# -----------------------------
# RSI (Força Relativa)
# -----------------------------
def add_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    # use explicit bfill assignment to avoid SettingWithCopyWarning
    data['rsi'] = data['rsi'].bfill()
    return data


# -----------------------------
# Lags - capturar autocorrelações (preço e retorno)
# - permite ao modelo LSTM aprender padrões temporais
# - empresta um contexto sobre a evolução no tempo
# -----------------------------
def create_lags(df, columns, n_lags):
    df_lags = df.copy()

    for col in columns:
        for lag in range(1, n_lags + 1):
            df_lags[f"{col}_lag{lag}"] = df_lags[col].shift(lag)

    df_lags = df_lags.dropna()  # remove NaN from the first rows
    return df_lags


# -----------------------------
# EMA
# -----------------------------
def add_moving_averages(df: pd.DataFrame, windows=[5, 12, 20, 26]) -> pd.DataFrame:
    if 'Close' not in df.columns:
        raise ValueError("O DataFrame precisa conter a coluna 'Close'.")
    df = df.copy()
    for w in windows:
        df[f'MA_{w}'] = df['Close'].rolling(window=w).mean()
        df[f'EMA_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()
    
    df['Spread_EMA'] = df['EMA_12'] - df['EMA_26']  # Diferença entre EMAs de 12 e 26 períodos - semelhante ao MACD
    
    # Média Movel Exponencial (9 períodos do Spread_EMA -> crossover com menor ruido)
    df['MACD_Signal_Line'] = df['Spread_EMA'].ewm(span=9, adjust=False).mean()
    
    # Histograma (diferença entre MACD e Signal Line -> aceleração de tendência) 
    df['MACD_Histograma'] = df['Spread_EMA'] - df['MACD_Signal_Line']

    df = df.dropna()
    print(df.head())
    return df


# -----------------------------
# LSTM data preparation
# -----------------------------
def prepare_lstm_data(data, window_size=30):
    # include RSI as an additional feature
    #features = data[['Close', 'volatility', 'rsi', 'Close_lag1']].values
    features = data.values
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    X, y = [], []
    for i in range(window_size, len(features_scaled)):
        X.append(features_scaled[i - window_size:i])
        y.append(features_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    return X, y, scaler


# -----------------------------
# LSTM Model Training
# -----------------------------
def train_lstm(X, y, epochs=50, batch_size=32):
    # Ensure correct dtypes and guard against NaNs
    X = X.astype('float32')
    y = y.astype('float32')

    print(f"Training data shape: X={X.shape}, y={y.shape}")
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("Training data contains NaNs. Please check data preprocessing steps.")

    model = Sequential([
        LSTM(50, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        Dense(1)
    ])

    # Use optimizer with gradient clipping to reduce risk of exploding gradients
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()

    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop, terminate_on_nan],
        verbose=1
    )

    # If training produced NaNs, warn the user
    loss_list = history.history.get('loss', [])
    if any([np.isnan(x) for x in loss_list]):
        print('Warning: training produced NaN loss. Consider lowering learning rate, using clipnorm, or inspecting the data for NaNs/infs.')

    return model, history


# -----------------------------
# Multiple predictions with noise
# -----------------------------
def predict_tomorrow(model, data, scaler, window_size=30, simulations=20):
    # include RSI when building the last window for prediction
    #last_features = data[['Close', 'volatility', 'rsi', 'Close_lag1']].values[-window_size:]
    last_features = data.values[-window_size:]
    last_scaled = scaler.transform(last_features)

    preds = []
    for _ in range(simulations):
        noise = np.random.normal(0, 0.01, last_scaled.shape)
        noisy_input = last_scaled + noise
        pred_scaled = model.predict(noisy_input.reshape(1, window_size, last_scaled.shape[1]), verbose=0)
        
        # inverse transform requires the same number of features used in fit_transform
        n_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else last_scaled.shape[1]
        zeros_count = n_features - 1
        pad = np.zeros((pred_scaled.shape[0], zeros_count))
        pred_price = scaler.inverse_transform(
            np.hstack([pred_scaled, pad])
        )[:, 0][0]
        preds.append(pred_price)

    mean_price = np.mean(preds)
    std_price = np.std(preds)
    today_close = float(data['Close'].iloc[-1])
    trend = "🔺 Alta" if mean_price > today_close else "🔻 Baixa"
    #tomorrow_date = (data.index[-1] + timedelta(days=1)).strftime("%d/%m")
    tomorrow_date = (data.index[-1] + timedelta(days=1))

    return mean_price, std_price, trend, tomorrow_date


# -----------------------------
# Plot Multi-Window + Overfitting
# -----------------------------
def plot_full_windows(data, windows=[30,45,60,90], simulations=20, epochs=50):
    fig, axes = plt.subplots(2, 4, figsize=(24,10))
    axes = axes.flatten()

    results = []  # armazenar métricas por janela

    for i, window_size in enumerate(windows):
        X, y, scaler = prepare_lstm_data(data, window_size)
        model, history = train_lstm(X, y, epochs=epochs)

        # Previsão de amanhã
        mean_price, std_price, trend, tomorrow_date = predict_tomorrow(model, data, scaler, window_size, simulations)

        X_for_pred = X
        pred_scaled = model.predict(X_for_pred, verbose=0)

        # pad predicted column with zeros for other features before inverse-scaling
        n_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else X.shape[2]
        zeros_count = n_features - 1
        pad = np.zeros((pred_scaled.shape[0], zeros_count))
        pred_price = scaler.inverse_transform(np.hstack([pred_scaled, pad]))[:,0]
        y_true = scaler.inverse_transform(np.hstack([y.reshape(-1, 1), np.zeros((y.shape[0], n_features - 1))]))[:, 0]

        # --- Métricas de desempenho ---
        mse = mean_squared_error(y_true, pred_price)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, pred_price)
        r2 = r2_score(y_true, pred_price)
        mape = np.mean(np.abs((y_true - pred_price) / y_true)) * 100

        results.append({
            "window": window_size,
            "price": mean_price,
            "date": tomorrow_date,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R2": r2
        })

        # Align predicted series x-axis to the last len(pred_price) dates
        pred_x = data.index[-len(pred_price):]

        # Linha 1: previsões
        ax = axes[i]
        ax.plot(data.index[-len(pred_price):], data['Close'][-len(pred_price):], label='Actual', color='blue')
        
        # Only plot if predictions are finite numbers
        if np.isfinite(pred_price).all():
            ax.plot(pred_x, pred_price, label='Predicted', color='orange')
        else:
            print(f"Warning: pred_price contains non-finite values: {pred_price}")

        ax.axhline(mean_price, color='green' if "Alta" in trend else 'red', linestyle='--', label=f'Tomorrow {trend}')
        ax.fill_between(data.index[-window_size:], mean_price-std_price, mean_price+std_price, color='gray', alpha=0.2)
        ax.text(
            data.index[-1],                # posição x (último dia disponível)
            mean_price,                    # posição y (nível da previsão)
            f'{tomorrow_date.strftime("%d/%m")} = {mean_price:.2f}',  # texto
            color='black',
            fontsize=10,
            va='bottom',                   # alinhamento vertical
            ha='left',                     # alinhamento horizontal
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
        )
        ax.set_title(f'Window {window_size}d\nRMSE={rmse:.2f} | MAE={mae:.2f} | R²={r2:.2f}',fontsize=11)
        ax.legend()
        ax.grid(True)

        # Linha 2: overfitting
        ax2 = axes[i+4]
        ax2.plot(history.history['loss'], label='Train Loss', color='blue')
        ax2.plot(history.history['val_loss'], label='Validation Loss', color='orange')
        ax2.set_title(f'Window {window_size} - Overfitting Check')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('MSE Loss')
        ax2.legend()
        ax2.grid(True)


    # Matriz de correlação completa
    corr = data.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Matriz de Correlação")


    plt.tight_layout()
    plt.show()
    return results

def check_colinearity(df, features, plot=True, threshold=0.9):
    corr = df[features].corr()
    
    if plot:
        plt.figure(figsize=(12,8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Matriz de Correlação")
        plt.show()
    
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i,j]))
    
    return high_corr


# -----------------------------
# Main pipeline
# -----------------------------
def main(args):
    data = download_data(args.ticker, start=args.start, end=args.end)
    data = add_weight_decay(data, weight=0.01)
    data = add_rolling_z_score(data, windows=[30, 45, 60, 90])
    data = get_vix_returns(data, start=args.start, end=args.end) 
    data = calculate_volatility(data)
    data = add_rsi(data)
    data = create_lags(data, ['Close', 'returns'], n_lags=5)
    data = add_moving_averages(data)

    features = ['Close', 'volatility', 'rsi'] + [f'Close_lag{i}' for i in range(1,6)] + [f'returns_lag{i}' for i in range(1,6)] + [f'MA_{w}' for w in [5,20]] + [f'EMA_{w}' for w in [12,26]] + ['Spread_EMA'] + ['MACD_Signal_Line'] + ['MACD_Histograma'] + ['VIX_Returns'] + [f'z{w}' for w in [60]] + ['sw']


    data = data[features]
    print(f"feature filtered dataset: {data.head()}")
    
    if getattr(args, 'feature_engineering', False):  
        high_corr_pairs = check_colinearity(data, features)
        if high_corr_pairs:
            print("\nAtenção: pares de features altamente correlacionadas:\n")
        for (f1, f2, corr) in high_corr_pairs:
            print(f"  - {f1[0] or f1[1]:<15} ↔ {f2[0] or f2[1]:<15} | correlação = {corr:.4f}")
    else:
        results = plot_full_windows(data, windows=[30, 45, 50, 75], simulations=args.simulations, epochs=args.epochs)
        save_results_to_json(results, filename="results/lstm_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid stock price prediction using LSTM + GARCH")
    parser.add_argument("--ticker", type=str, default="VALE3.SA", help="Stock ticker")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--simulations", type=int, default=20, help="Number of simulations for noise averaging")
    parser.add_argument("--feature-engineering",action="store_true", help="Enable feature-engineering checks (e.g. colinearity heatmap). Default: False")
    args = parser.parse_args()

    main(args)
