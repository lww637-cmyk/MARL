# Task for Claude Code

Context: this is my MARL/FinTech research project (PettingZoo env, two PPO agents,
Gemini coordinator). My sentiment data has multi-year gaps because the source news file
was over-filtered. Do these steps in order and stop at the end for my review.

## STEP 1 - Data extraction (read-only on my data)
Ask me for the full path of the large news file, then run:
python extract_news.py "<that path>" data/raw/news_headlines.csv
Show me the coverage report (headlines per ticker, % business days covered, longest gap).
Do not proceed to later steps if any ticker has a longest gap over 10 days; ask me first.

## STEP 2 - Code fixes (do not retrain anything)
a) src/data/fetch_sentiment.py: make load_news default to data/raw/news_headlines.csv.
b) src/environment/stock_env.py, build_merged_dataset: replace the unlimited ffill()
   with ffill(limit=5) on sentiment columns, then fillna(0.0). has_sentiment must be 1
   only on days with real headlines.
c) Add a chronological split: train = dates up to 2022-12-31, test = 2023-01-01 onward.
   The env takes a split argument. Training uses only the train split with random starts.
   Evaluation uses only the test split.
d) evaluate_agents.py: make buy_and_hold_multi_seed use exactly the same start windows
   (same seeds, same episode length, test split only) as the MARL runs, so the
   comparison is like-for-like.
e) Keep all other behaviour unchanged. Summarise every diff.

## STEP 3 - Sanity checks
Run a quick check (no training) that: sentiment coverage per ticker in the merged
dataset is sensible, has_sentiment matches real headline days, and train/test dates
do not overlap. Report results.

When committing, use clear messages. Do not push.
