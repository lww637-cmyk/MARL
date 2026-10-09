import pandas as pd

tickers = ['AAPL', 'AMZN', 'MSFT', 'NVDA', 'TSLA']

print('=== PRICE DATA ===')
for t in tickers:
    df = pd.read_csv(f'data/processed/{t}_processed.csv',
                     parse_dates=['date'], index_col='date')
    print(f'{t}: {len(df)} days, {df.shape[1]} features')

print()
print('=== SENTIMENT DATA ===')
for t in tickers:
    df = pd.read_csv(f'data/sentiment/{t}_sentiment.csv',
                     parse_dates=['date'], index_col='date')
    print(f'{t}: {len(df)} days, mean score={df["sentiment_score"].mean():.3f}')

print()
print('Phase 1 complete.')