import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("""# 05 — FinBERT Sentiment Scoring (Google Colab)

Scores `data/raw/news_headlines.csv` with FinBERT on the GPU and writes
`data/sentiment/{TICKER}_sentiment.csv` for the tickers in `config.yaml`.

**Before running**
1. Runtime → Change runtime type → **T4 GPU**
2. Upload `data/raw/news_headlines.csv` to `MyDrive/MARL/data/raw/news_headlines.csv`
3. Push the repo so the clone below has the current `config.yaml` and `src/`"""),

    nbf.v4.new_markdown_cell("## Step 1 — Check GPU"),
    nbf.v4.new_code_cell("""import torch
assert torch.cuda.is_available(), "No GPU — set Runtime > Change runtime type > GPU"
print("Device:", torch.cuda.get_device_name(0))"""),

    nbf.v4.new_markdown_cell("## Step 2 — Mount Google Drive"),
    nbf.v4.new_code_cell("""from google.colab import drive
drive.mount("/content/drive")

DRIVE = "/content/drive/MyDrive/MARL"
print("Drive mounted.")"""),

    nbf.v4.new_markdown_cell("## Step 3 — Clone repo and install"),
    nbf.v4.new_code_cell("""import os

if not os.path.exists("/content/MARL"):
    !git clone https://github.com/lww637-cmyk/MARL.git /content/MARL
else:
    !git -C /content/MARL pull
%cd /content/MARL
!git log --oneline -1"""),
    nbf.v4.new_code_cell("""!pip install -q -e .
!pip install -q transformers

import sys
sys.path.insert(0, "/content/MARL")
print("Installed.")"""),

    nbf.v4.new_markdown_cell("## Step 4 — Copy headlines from Drive"),
    nbf.v4.new_code_cell("""import shutil

os.makedirs("data/raw", exist_ok=True)
shutil.copy(f"{DRIVE}/data/raw/news_headlines.csv", "data/raw/news_headlines.csv")
print("Copied news_headlines.csv")"""),

    nbf.v4.new_markdown_cell("## Step 5 — Score with FinBERT on GPU, aggregate daily, save\n"
                             "Writes `data/sentiment/{TICKER}_sentiment.csv`."),
    nbf.v4.new_code_cell("""from src.data.fetch_sentiment import load_config, score_and_save

config  = load_config()
TICKERS = config["data"]["tickers"]
print("Tickers:", TICKERS)

score_and_save(config, batch_size=128, device=0)"""),

    nbf.v4.new_markdown_cell("## Step 6 — Copy sentiment files to Drive\n"
                             "Existing files on Drive are kept as `*_sentiment_old.csv`."),
    nbf.v4.new_code_cell("""os.makedirs(f"{DRIVE}/data/sentiment", exist_ok=True)
for t in TICKERS:
    dst = f"{DRIVE}/data/sentiment/{t}_sentiment.csv"
    if os.path.exists(dst):
        shutil.copy(dst, f"{DRIVE}/data/sentiment/{t}_sentiment_old.csv")
    shutil.copy(f"data/sentiment/{t}_sentiment.csv", dst)
    print(f"Saved {t} → {dst}")"""),

    nbf.v4.new_markdown_cell("## Step 7 — Check: headline days per ticker per year\n"
                             "Calendar days with at least one headline (before rolling weekends to trading days)."),
    nbf.v4.new_code_cell("""import pandas as pd

days = {}
for t in TICKERS:
    df = pd.read_csv(f"data/sentiment/{t}_sentiment.csv", parse_dates=["date"])
    days[t] = df.groupby(df["date"].dt.year).size()

table = pd.DataFrame(days).fillna(0).astype(int)
table.index.name = "year"
print(table.to_string())
print()
for t in TICKERS:
    df = pd.read_csv(f"data/sentiment/{t}_sentiment.csv")
    print(f"{t}: mean score {df['sentiment_score'].mean():+.3f} | "
          f"headlines {int(df['headline_count'].sum())}")"""),
]

nb.cells = cells
nb.metadata["accelerator"] = "GPU"
nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
nb.metadata["kernelspec"] = {"display_name": "Python 3", "name": "python3"}

with open("notebooks/05_colab_score_sentiment.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook created → notebooks/05_colab_score_sentiment.ipynb")
