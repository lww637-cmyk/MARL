# TASK2: switch the project to new tickers (do steps in order, stop at the end)

Reason: the source news dataset has long holes for AAPL, AMZN, MSFT, NVDA, TSLA.
New tickers (selection rule: highest news coverage per sector, longest gap <= 13 days,
>= 85% business days covered): INTC, AMD, GS, CVX, SBUX.
News in the source ends on 2023-12-16, so the study end date becomes 2023-12-15.

## 1. Build the news file
From data/raw/candidates.csv keep only INTC, AMD, GS, CVX, SBUX and dates
2018-07-25 to 2023-12-15. Save as data/raw/news_headlines.csv (columns date,
article_title, stock_symbol). Keep the old file as data/raw/news_headlines_old5.csv.
Print headlines per ticker, % business days covered, and the longest gap per ticker
(edges included, within 2018-07-25..2023-12-15).

## 2. Replace tickers everywhere
- config.yaml: tickers = INTC, AMD, GS, CVX, SBUX; start_date 2018-07-25; end_date 2023-12-15.
- grep the whole repo (src, notebooks scripts, tests, dashboard, SHAP, evaluate, create_results_notebook.py)
  for hard-coded AAPL, AMZN, MSFT, NVDA, TSLA, GOOGL and replace them from config
  (or the TICKERS constant). Report every place you changed.
- Keep the split: train to 2022-06-30, test from 2022-07-01 to the end of data.
  Confirm the test split still has at least episode_length (252) trading days.

## 3. New price data
- Re-download adjusted OHLCV for the 5 tickers from yfinance for 2018-07-25..2023-12-15
  (flatten MultiIndex columns), and rebuild the technical indicators exactly as before.
- Save to the same data/ locations the pipeline expects. Keep old price files in a backup folder.
- Report rows per ticker and any date misalignment between tickers.

## 4. Checks (no training, no FinBERT yet)
- Sentiment merge: with news_headlines.csv only (not yet scored), confirm the pipeline
  runs end to end and the observation sizes match what the agents expect (print them).
- Re-run test_env.py.
- Normalise headline_count with log1p then the TRAIN-period maximum only; roll weekend and
  holiday headlines forward to the next trading day (approved earlier). Verify with a tiny example.
- Commit in small commits. Do not push. Do not retrain.

Report: list of files changed, the coverage table, observation sizes, test results.
