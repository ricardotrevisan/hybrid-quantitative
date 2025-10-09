import yfinance as yf
import pandas as pd
import mplfinance as mpf

tickers = ['VALE3.SA']
tickers = ['PETR4.SA']
# Baixar dados históricos
df = yf.download(tickers[0], period='6mo', interval='1d', auto_adjust=True)

print(df.head())
print(df.columns)
print(df[0:5])

# Achatar MultiIndex, se existir
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

# Converter OHLC e Volume para float e remover NaNs
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col] = df[col].astype(float)

df = df.dropna(subset=['Open','High','Low','Close','Volume'])

# Garantir datetime no índice
df.index = pd.to_datetime(df.index)

# Calcular EMAs
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

# MACD e Signal Line
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['Histograma'] = df['MACD'] - df['Signal']

# Sinais de compra e venda
df['Sinal_Compra'] = ((df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1)))
df['Sinal_Venda'] = ((df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1)))

# Preparar addplots
apds = [
    mpf.make_addplot(df['MACD'], panel=1, color='blue', ylabel='MACD'),
    mpf.make_addplot(df['Signal'], panel=1, color='red'),
    mpf.make_addplot(df['Histograma'], type='bar', panel=1, color='green', alpha=0.5)
]

# Corrigir sinais para tamanho igual ao DataFrame
buy_signals = df['Close'].where(df['Sinal_Compra'])
sell_signals = df['Close'].where(df['Sinal_Venda'])

if not buy_signals.isna().all():
    apds.append(mpf.make_addplot(buy_signals, type='scatter', markersize=100, marker='^', color='green'))
if not sell_signals.isna().all():
    apds.append(mpf.make_addplot(sell_signals, type='scatter', markersize=100, marker='v', color='red'))

# Plotar candles com MACD
mpf.plot(df, type='candle', style='charles', addplot=apds, volume=True, figsize=(14,8), title=f'{tickers[0]} com MACD e Sinais')
