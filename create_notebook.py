import nbformat as nbf
import json

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# 02 — Sentiment Scoring (Run on Google Colab)\n\nThis notebook runs FinBERT on 43,485 news headlines to produce daily company-specific sentiment scores.\n**Run this entire notebook on Google Colab with GPU enabled.**\n\nRuntime → Change runtime type → T4 GPU"),

    nbf.v4.new_markdown_cell("## Step 1 — Check GPU"),

    nbf.v4.new_code_cell("""import torch
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"""),

    nbf.v4.new_markdown_cell("## Step 2 — Mount Google Drive"),

    nbf.v4.new_code_cell("""from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/drive/MyDrive/MARL/data/sentiment', exist_ok=True)
print('Drive mounted and folders ready.')"""),

    nbf.v4.new_markdown_cell("## Step 3 — Clone your GitHub repo\nReplace `yourusername` with your actual GitHub username."),

    nbf.v4.new_code_cell("""!git clone https://github.com/yourusername/MARL.git
%cd MARL
!pip install -r requirements.txt -q"""),

    nbf.v4.new_markdown_cell("## Step 4 — Upload news file\nUpload your `news_filtered.csv` when prompted. This saves it to the correct location inside the repo."),

    nbf.v4.new_code_cell("""from google.colab import files
import shutil

print('Upload your news_filtered.csv file...')
uploaded = files.upload()

for filename in uploaded.keys():
    shutil.move(filename, 'data/raw/news_filtered.csv')
    print(f'Moved {filename} to data/raw/news_filtered.csv')"""),

    nbf.v4.new_markdown_cell("## Step 5 — Validate news file"),

    nbf.v4.new_code_cell("""import sys
sys.path.insert(0, '/content/MARL')

from src.data.fetch_sentiment import load_news, load_config

config = load_config()
tickers = config['data']['tickers']
df = load_news(tickers=tickers)
print('\\nNews file validated successfully.')
print(df[['date', 'ticker', 'article_title']].head(3).to_string())"""),

    nbf.v4.new_markdown_cell("## Step 6 — Run FinBERT scoring\nThis takes ~15-20 minutes on T4 GPU."),

    nbf.v4.new_code_cell("""from src.data.fetch_sentiment import score_and_save
score_and_save(config)"""),

    nbf.v4.new_markdown_cell("## Step 7 — Verify output files"),

    nbf.v4.new_code_cell("""import pandas as pd
import os

for ticker in tickers:
    path = f'data/sentiment/{ticker}_sentiment.csv'
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['date'], index_col='date')
        print(f'{ticker}: {len(df)} days | score range: {df[\"sentiment_score\"].min():.3f} to {df[\"sentiment_score\"].max():.3f}')
    else:
        print(f'WARNING: {path} not found')"""),

    nbf.v4.new_markdown_cell("## Step 8 — Copy results to Google Drive"),

    nbf.v4.new_code_cell("""import shutil

for ticker in tickers:
    src = f'data/sentiment/{ticker}_sentiment.csv'
    dst = f'/content/drive/MyDrive/MARL/data/sentiment/{ticker}_sentiment.csv'
    shutil.copy(src, dst)
    print(f'Copied {ticker}_sentiment.csv to Google Drive')

print('\\nAll sentiment files saved to Google Drive.')"""),

    nbf.v4.new_markdown_cell("## Step 9 — Download files to your laptop"),

    nbf.v4.new_code_cell("""from google.colab import files

for ticker in tickers:
    path = f'data/sentiment/{ticker}_sentiment.csv'
    files.download(path)
    print(f'Downloaded {ticker}_sentiment.csv')"""),
]

nb.cells = cells
output_path = "notebooks/02_feature_engineering.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbformat_str = nbf.writes(nb)
    f.write(nbformat_str)

print(f"Notebook created at {output_path}")