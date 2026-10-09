import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("""# 01 — Exploratory Data Analysis
Explores price and sentiment data for AAPL, AMZN, MSFT, NVDA, TSLA (2018-2023).
Run locally after Phase 1 data collection is complete."""),

    nbf.v4.new_markdown_cell("## Setup"),

    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Plot style
plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

TICKERS = ['AAPL', 'AMZN', 'MSFT', 'NVDA', 'TSLA']
COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
print('Setup complete.')"""),

    nbf.v4.new_markdown_cell("## 1. Load Price Data"),

    nbf.v4.new_code_cell("""import sys
sys.path.insert(0, '..')

from src.data.fetch_price import load_all, split_by_ticker, filter_date_range
from src.data.preprocess import load_config

config = load_config()
raw = load_all('../data/raw/prices_all.csv')
raw = filter_date_range(raw, config['data']['start_date'], config['data']['end_date'])
ticker_data = split_by_ticker(raw)

print('Tickers loaded:', list(ticker_data.keys()))
print('\\nAAPL sample:')
ticker_data['AAPL'].head()"""),

    nbf.v4.new_markdown_cell("## 2. Basic Price Statistics"),

    nbf.v4.new_code_cell("""summary_rows = []
for ticker in TICKERS:
    df = ticker_data[ticker]
    summary_rows.append({
        'Ticker'     : ticker,
        'Start'      : df.index.min().date(),
        'End'        : df.index.max().date(),
        'Trading Days': len(df),
        'Min Close'  : round(df['close'].min(), 2),
        'Max Close'  : round(df['close'].max(), 2),
        'Mean Close' : round(df['close'].mean(), 2),
        'Std Close'  : round(df['close'].std(), 2),
        'Missing'    : df['close'].isna().sum()
    })

summary = pd.DataFrame(summary_rows).set_index('Ticker')
print(summary.to_string())"""),

    nbf.v4.new_markdown_cell("## 3. Price Trends (2018-2023)"),

    nbf.v4.new_code_cell("""fig, ax = plt.subplots()
for ticker, color in zip(TICKERS, COLORS):
    df = ticker_data[ticker]
    # Normalise to 100 at start for fair comparison
    normalised = df['close'] / df['close'].iloc[0] * 100
    ax.plot(df.index, normalised, label=ticker, color=color, linewidth=1.5)

ax.set_title('Normalised Close Price (Base=100)', fontsize=14)
ax.set_ylabel('Normalised Price')
ax.set_xlabel('Date')
ax.legend()

# Annotate key market events
events = {
    'COVID Crash': '2020-03-20',
    'Rate Hikes': '2022-01-01',
    'AI Boom': '2023-01-01'
}
for label, date in events.items():
    ax.axvline(pd.Timestamp(date), color='gray', linestyle='--', alpha=0.6)
    ax.text(pd.Timestamp(date), ax.get_ylim()[1]*0.95, label,
            rotation=90, fontsize=8, color='gray', va='top')

plt.tight_layout()
plt.show()"""),

    nbf.v4.new_markdown_cell("## 4. Daily Returns Distribution"),

    nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, ticker, color in zip(axes, TICKERS, COLORS):
    returns = ticker_data[ticker]['close'].pct_change().dropna()
    ax.hist(returns, bins=60, color=color, alpha=0.75, edgecolor='white')
    ax.set_title(ticker)
    ax.set_xlabel('Daily Return')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.text(0.05, 0.92, f'Mean: {returns.mean():.4f}\\nStd:  {returns.std():.4f}\\nSkew: {returns.skew():.2f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

fig.suptitle('Daily Return Distributions', fontsize=14)
plt.tight_layout()
plt.show()"""),

    nbf.v4.new_markdown_cell("## 5. Volatility Analysis (30-day Rolling Std)"),

    nbf.v4.new_code_cell("""fig, ax = plt.subplots()
for ticker, color in zip(TICKERS, COLORS):
    returns = ticker_data[ticker]['close'].pct_change()
    rolling_vol = returns.rolling(30).std() * np.sqrt(252)
    ax.plot(ticker_data[ticker].index, rolling_vol,
            label=ticker, color=color, linewidth=1.2)

ax.set_title('30-Day Rolling Annualised Volatility', fontsize=14)
ax.set_ylabel('Annualised Volatility')
ax.set_xlabel('Date')
ax.legend()

for label, date in events.items():
    ax.axvline(pd.Timestamp(date), color='gray', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()"""),

    nbf.v4.new_markdown_cell("## 6. Ticker Correlation Matrix"),

    nbf.v4.new_code_cell("""returns_df = pd.DataFrame({
    ticker: ticker_data[ticker]['close'].pct_change()
    for ticker in TICKERS
}).dropna()

corr = returns_df.corr()

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)

ax.set_xticks(range(len(TICKERS)))
ax.set_yticks(range(len(TICKERS)))
ax.set_xticklabels(TICKERS)
ax.set_yticklabels(TICKERS)

for i in range(len(TICKERS)):
    for j in range(len(TICKERS)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                ha='center', va='center', fontsize=11,
                color='black')

ax.set_title('Return Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.show()

print('\\nKey insight: highly correlated tickers may not add much diversity across agents.')"""),

    nbf.v4.new_markdown_cell("## 7. Load Processed Data (with Technical Indicators)"),

    nbf.v4.new_code_cell("""processed = {}
for ticker in TICKERS:
    path = f'../data/processed/{ticker}_processed.csv'
    processed[ticker] = pd.read_csv(path, parse_dates=['date'], index_col='date')

print('Processed data loaded.')
print('Columns:', list(processed['AAPL'].columns))
processed['AAPL'].head(3)"""),

    nbf.v4.new_markdown_cell("## 8. Technical Indicators Overview (AAPL example)"),

    nbf.v4.new_code_cell("""ticker = 'AAPL'
df = processed[ticker].copy()

fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

# Close price + Bollinger Bands
axes[0].plot(df.index, df['close'], label='Close', color='#1f77b4')
axes[0].plot(df.index, df['bb_upper'] * df['close'], label='BB Upper',
             color='gray', linestyle='--', alpha=0.7)
axes[0].plot(df.index, df['bb_lower'] * df['close'], label='BB Lower',
             color='gray', linestyle='--', alpha=0.7)
axes[0].set_title(f'{ticker} — Close Price with Bollinger Bands')
axes[0].legend()

# RSI
axes[1].plot(df.index, df['rsi'] * 100, color='purple', linewidth=1)
axes[1].axhline(70, color='red', linestyle='--', alpha=0.6, label='Overbought (70)')
axes[1].axhline(30, color='green', linestyle='--', alpha=0.6, label='Oversold (30)')
axes[1].set_title('RSI (14)')
axes[1].set_ylabel('RSI')
axes[1].legend()

# MACD
axes[2].plot(df.index, df['macd'], label='MACD', color='blue', linewidth=1)
axes[2].plot(df.index, df['macd_signal'], label='Signal', color='orange', linewidth=1)
axes[2].bar(df.index, df['macd_diff'], label='Histogram',
            color=['green' if x >= 0 else 'red' for x in df['macd_diff']], alpha=0.4)
axes[2].set_title('MACD')
axes[2].legend()

# Daily return
axes[3].bar(df.index, df['daily_return'],
            color=['green' if x >= 0 else 'red' for x in df['daily_return']], alpha=0.6)
axes[3].set_title('Daily Return')
axes[3].set_ylabel('Return')

plt.tight_layout()
plt.show()"""),

    nbf.v4.new_markdown_cell("## 9. Sentiment EDA (run after FinBERT scoring is complete)"),

    nbf.v4.new_code_cell("""import os

sentiment = {}
missing = []
for ticker in TICKERS:
    path = f'../data/sentiment/{ticker}_sentiment.csv'
    if os.path.exists(path):
        sentiment[ticker] = pd.read_csv(path, parse_dates=['date'], index_col='date')
        print(f'{ticker}: {len(sentiment[ticker])} days loaded')
    else:
        missing.append(ticker)
        print(f'{ticker}: NOT FOUND — run FinBERT scoring first')

if missing:
    print(f'\\nSkipping sentiment plots for: {missing}')"""),

    nbf.v4.new_markdown_cell("## 10. Sentiment Score Distribution"),

    nbf.v4.new_code_cell("""if sentiment:
    fig, axes = plt.subplots(1, len(sentiment), figsize=(18, 4))
    if len(sentiment) == 1:
        axes = [axes]
    for ax, (ticker, df) in zip(axes, sentiment.items()):
        ax.hist(df['sentiment_score'], bins=40,
                color=COLORS[TICKERS.index(ticker)], alpha=0.75, edgecolor='white')
        ax.axvline(0, color='black', linewidth=1)
        ax.set_title(ticker)
        ax.set_xlabel('Sentiment Score')
        mean_val = df['sentiment_score'].mean()
        ax.text(0.05, 0.92, f'Mean: {mean_val:.3f}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
    fig.suptitle('Daily Sentiment Score Distributions', fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    print('No sentiment data available yet.')"""),

    nbf.v4.new_markdown_cell("## 11. Sentiment vs Price (overlay)"),

    nbf.v4.new_code_cell("""if sentiment:
    for ticker in list(sentiment.keys())[:2]:  # Show first 2 tickers
        price = processed[ticker]['close']
        sent  = sentiment[ticker]['sentiment_score']

        # Align on common dates
        common = price.index.intersection(sent.index)
        price_aligned = price.loc[common]
        sent_aligned  = sent.loc[common]

        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax2 = ax1.twinx()

        ax1.plot(price_aligned.index, price_aligned,
                 color='#1f77b4', linewidth=1.5, label='Close Price')
        ax2.fill_between(sent_aligned.index, sent_aligned, 0,
                         where=sent_aligned >= 0, alpha=0.3,
                         color='green', label='Positive Sentiment')
        ax2.fill_between(sent_aligned.index, sent_aligned, 0,
                         where=sent_aligned < 0, alpha=0.3,
                         color='red', label='Negative Sentiment')

        ax1.set_title(f'{ticker} — Close Price vs Daily Sentiment Score', fontsize=13)
        ax1.set_ylabel('Close Price', color='#1f77b4')
        ax2.set_ylabel('Sentiment Score')
        ax2.axhline(0, color='gray', linewidth=0.8)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        plt.show()
else:
    print('No sentiment data available yet.')"""),

    nbf.v4.new_markdown_cell("## 12. Sentiment Coverage — Missing Trading Days"),

    nbf.v4.new_code_cell("""if sentiment:
    price_dates = ticker_data['AAPL'].index.normalize()

    print(f'Total trading days in price data: {len(price_dates)}')
    print()
    for ticker, df in sentiment.items():
        sent_dates    = df.index.normalize()
        covered       = len(price_dates.intersection(sent_dates))
        missing_count = len(price_dates) - covered
        pct           = covered / len(price_dates) * 100
        print(f'{ticker}: {covered} days covered ({pct:.1f}%), {missing_count} trading days with no news')
else:
    print('No sentiment data available yet.')"""),

    nbf.v4.new_markdown_cell("## 13. Sentiment-Return Cross Correlation (lag analysis)"),

    nbf.v4.new_code_cell("""if sentiment:
    print('Cross-correlation: Sentiment Score vs Next-Day Return')
    print('(positive lag = sentiment leads price)\\n')

    for ticker in list(sentiment.keys()):
        price   = processed[ticker]['daily_return']
        sent    = sentiment[ticker]['sentiment_score']
        common  = price.index.intersection(sent.index)

        p = price.loc[common]
        s = sent.loc[common]

        lags = range(-5, 6)
        corrs = [p.corr(s.shift(lag)) for lag in lags]

        best_lag  = lags[np.argmax(np.abs(corrs))]
        best_corr = corrs[np.argmax(np.abs(corrs))]
        print(f'{ticker}: best lag = {best_lag} days, correlation = {best_corr:.4f}')
else:
    print('No sentiment data available yet.')"""),

    nbf.v4.new_markdown_cell("## 14. EDA Summary & Key Findings"),

    nbf.v4.new_code_cell("""print(\"=\"*60)
print(\"EDA SUMMARY\")
print(\"=\"*60)

print(\"\\n--- Price Data ---\")
for ticker in TICKERS:
    df = ticker_data[ticker]
    returns = df['close'].pct_change().dropna()
    total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    annual_vol   = returns.std() * np.sqrt(252) * 100
    print(f\"{ticker}: Total return = {total_return:.1f}%, Annual vol = {annual_vol:.1f}%\")

if sentiment:
    print(\"\\n--- Sentiment Data ---\")
    for ticker, df in sentiment.items():
        pos = (df['sentiment_score'] > 0.1).mean() * 100
        neg = (df['sentiment_score'] < -0.1).mean() * 100
        neu = 100 - pos - neg
        print(f\"{ticker}: Positive={pos:.1f}%, Neutral={neu:.1f}%, Negative={neg:.1f}%\")

print(\"\\nEDA complete. Proceed to environment setup (Phase 2).\")"""),
]

nb.cells = cells

output_path = "notebooks/01_data_exploration.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    import nbformat as nbf_module
    f.write(nbf_module.writes(nb))

print(f"EDA notebook created at {output_path}")