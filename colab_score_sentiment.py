"""
Colab cells: re-score data/raw/news_headlines.csv with FinBERT on GPU and
write data/sentiment/{TICKER}_sentiment.csv back to Google Drive.

Before running:
  1. Runtime > Change runtime type > GPU.
  2. Upload data/raw/news_headlines.csv to
     MyDrive/MARL/data/raw/news_headlines.csv
  3. Push this repo so the clone below contains the updated
     src/data/fetch_sentiment.py (load_news default + device argument).

Paste each "# %%" block into its own Colab cell, or run the whole file
with `%run colab_score_sentiment.py` from the repo root after cloning.
"""

# %% 1. GPU check
import torch
assert torch.cuda.is_available(), "No GPU — set Runtime > Change runtime type > GPU"
print("Device:", torch.cuda.get_device_name(0))

# %% 2. Mount Drive, clone repo, install deps
import os
import shutil
from google.colab import drive

drive.mount("/content/drive")
DRIVE = "/content/drive/MyDrive/MARL"

if not os.path.exists("/content/MARL"):
    os.system("git clone https://github.com/lww637-cmyk/MARL.git /content/MARL")
os.chdir("/content/MARL")
os.system("pip install -q transformers pandas pyyaml")

# %% 3. Copy headlines into the repo
os.makedirs("data/raw", exist_ok=True)
shutil.copy(f"{DRIVE}/data/raw/news_headlines.csv", "data/raw/news_headlines.csv")
print("Copied news_headlines.csv")

# %% 4. Score with FinBERT on GPU, aggregate daily, save
import sys
sys.path.insert(0, "/content/MARL")
from src.data.fetch_sentiment import load_config, score_and_save

config = load_config()
score_and_save(config, batch_size=128, device=0)

# %% 5. Copy sentiment files back to Drive (old ones kept as *_sentiment_old.csv)
os.makedirs(f"{DRIVE}/data/sentiment", exist_ok=True)
for t in config["data"]["tickers"]:
    dst = f"{DRIVE}/data/sentiment/{t}_sentiment.csv"
    if os.path.exists(dst):
        shutil.copy(dst, f"{DRIVE}/data/sentiment/{t}_sentiment_old.csv")
    shutil.copy(f"data/sentiment/{t}_sentiment.csv", dst)
    print(f"Saved {t} → {dst}")
