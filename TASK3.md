# TASK3: prepare for Colab scoring and retraining (do in order, stop at the end, do not push)

## 1. Two fixes in src/environment/stock_env.py (build_merged_dataset)
a) headline_count must be 0.0 on days with no real headlines. Do NOT forward-fill the count.
   Sentiment score and std may still carry forward up to 5 trading days (has_sentiment stays 0 on those days).
b) Clip headline_count_norm to [0, 1] after log1p / train-max scaling.
Update test_sentiment_merge.py to cover both, and run test_env.py + test_sentiment_merge.py.

## 2. Housekeeping
- Commit extract_news.py as it is now.
- Check .gitignore. Tell me exactly which data files Colab needs (news_headlines.csv, processed
  price CSVs, sentiment CSVs) and how each one reaches Colab (GitHub vs Google Drive). Do not
  commit large data files.

## 3. Colab scoring notebook
Convert colab_score_sentiment.py into notebooks/05_colab_score_sentiment.ipynb using nbformat
(open files with encoding="utf-8", real separate cells, not one cell). It should:
mount Drive, clone the repo, pip install -e ., install transformers, read news_headlines.csv from
MyDrive/MARL/data/raw/, score with FinBERT on the GPU, write data/sentiment/{TICKER}_sentiment.csv,
copy them back to MyDrive/MARL/data/sentiment/, and print headline days per ticker per year.

## 4. Colab training notebook
Check the training notebook: it must (a) get the NEW price files and sentiment files, (b) accept a
--seed value, (c) save checkpoints to results/checkpoints/seedN/ and copy them to Drive after every
100k steps so a Colab disconnect does not lose work, (d) train seed 0 first and stop, so I can check
the policy before spending more GPU time. Report any gap between this and what exists.

## 5. Report
List files changed, test results, and the exact manual steps I must do (what to push, what to upload
to Drive, which notebook to open and its GitHub URL). Do not push and do not train.
