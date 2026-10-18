import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("""# 03 — MARL Training (Google Colab)

Trains the price agent and the sentiment agent for **one training seed** (`SEED`, default 0)
for `training.total_timesteps` steps each (see `config.yaml`).

Checkpoints are saved every `training.save_frequency` steps to `results/checkpoints/seed{SEED}/`
and copied to `MyDrive/MARL/results/checkpoints/seed{SEED}/` as soon as each one is written.

**Before running**
1. Push the repo (current `config.yaml`, `src/`).
2. On Drive: `MyDrive/MARL/data/processed/{TICKER}_processed.csv` and
   `MyDrive/MARL/data/sentiment/{TICKER}_sentiment.csv` (written by notebook 05) for every ticker.

Train seed 0 first, run the policy check at the end, and stop. Change `SEED` and re-run only after checking."""),

    nbf.v4.new_markdown_cell("## Step 1 — Training seed"),
    nbf.v4.new_code_cell('SEED = 0'),

    nbf.v4.new_markdown_cell("## Step 2 — Mount Google Drive"),
    nbf.v4.new_code_cell('from google.colab import drive\ndrive.mount("/content/drive")\n\nDRIVE = "/content/drive/MyDrive/MARL"\nprint("Drive mounted.")'),

    nbf.v4.new_markdown_cell("## Step 3 — Clone repo and install dependencies"),
    nbf.v4.new_code_cell('import os\n\nif not os.path.exists("/content/MARL"):\n    !git clone https://github.com/lww637-cmyk/MARL.git /content/MARL\nelse:\n    !git -C /content/MARL pull\n%cd /content/MARL\n!git log --oneline -1'),
    nbf.v4.new_code_cell('!pip install -q -r requirements.txt\n!pip install -q -e .\n\nimport sys\nsys.path.insert(0, "/content/MARL")\nprint("Dependencies installed.")'),

    nbf.v4.new_markdown_cell("## Step 4 — Copy data files from Drive\nFails if any processed or sentiment file is missing, so training never runs without sentiment."),
    nbf.v4.new_code_cell('import shutil\nimport pandas as pd\nfrom src.training.train_marl import load_config\n\nconfig  = load_config()\nTICKERS = config["data"]["tickers"]\nprint("Tickers:", TICKERS, "| end_date:", config["data"]["end_date"])\n\nos.makedirs("data/processed", exist_ok=True)\nos.makedirs("data/sentiment", exist_ok=True)\n\nmissing = []\nfor t in TICKERS:\n    for kind in ["processed", "sentiment"]:\n        src = f"{DRIVE}/data/{kind}/{t}_{kind}.csv"\n        if not os.path.exists(src):\n            missing.append(src)\n            continue\n        shutil.copy(src, f"data/{kind}/{t}_{kind}.csv")\nassert not missing, f"Missing on Drive: {missing}"\n\nfor t in TICKERS:\n    p = pd.read_csv(f"data/processed/{t}_processed.csv", parse_dates=["date"])\n    s = pd.read_csv(f"data/sentiment/{t}_sentiment.csv", parse_dates=["date"])\n    print(f"{t}: prices {p.date.min().date()}..{p.date.max().date()} ({len(p)} rows) | "\n          f"sentiment {s.date.min().date()}..{s.date.max().date()} ({len(s)} days)")'),

    nbf.v4.new_markdown_cell("## Step 5 — Validate environment (train split)"),
    nbf.v4.new_code_cell('from src.environment.stock_env import StockTradingEnv\n\nenv = StockTradingEnv(config, mode="train", split="train")\nprint(f"Train split: {env.n_steps} trading days "\n      f"({env.common_dates[0].date()} .. {env.common_dates[-1].date()})")\nprint("Price obs:", env.observation_space("price_agent").shape,\n      "| Sentiment obs:", env.observation_space("sentiment_agent").shape)\n\nfor t in TICKERS:\n    d = env.dataset[t].loc[env.common_dates]\n    print(f"  {t}: has_sentiment on {d.has_sentiment.mean()*100:.1f}% of train days")\n    assert d.has_sentiment.mean() > 0, f"{t} has no sentiment in the train split"'),

    nbf.v4.new_markdown_cell("## Step 6 — Train one seed (resumes automatically)\nEvery checkpoint is copied to Drive as soon as it is saved. "
                             "If Colab disconnects, re-run the notebook from the top: each agent continues from "
                             "its newest checkpoint in `MyDrive/MARL/results/checkpoints/seed{SEED}/`, "
                             "and an agent with a `_final.zip` there is skipped."),
    nbf.v4.new_code_cell('from src.training.train_marl import find_checkpoints\n\nDRIVE_CKPT = f"{DRIVE}/results/checkpoints/seed{SEED}"\nTOTAL      = config["training"]["total_timesteps"]\n\nfor agent_id in ["price_agent", "sentiment_agent"]:\n    found = find_checkpoints(agent_id, [DRIVE_CKPT], TOTAL)\n    if found:\n        step, path = found[0]\n        print(f"{agent_id}: will resume from step {step:,} of {TOTAL:,} ({os.path.basename(path)})")\n    else:\n        print(f"{agent_id}: no checkpoint on Drive — starts from step 0")'),
    nbf.v4.new_code_cell('from src.training.train_marl import train\n\nprice_agent, sentiment_agent = train(config, seed=SEED, copy_dir=DRIVE_CKPT)\nprint("Training complete. Checkpoints on Drive:")\nprint(sorted(os.listdir(DRIVE_CKPT)))'),

    nbf.v4.new_markdown_cell("## Step 7 — Policy check (train split only), then STOP\n"
                             "Deterministic actions of each final model on every train-split day. "
                             "A per-asset std near 0 means the policy ignores its inputs "
                             "(e.g. a constant uniform allocation)."),
    nbf.v4.new_code_cell('import numpy as np\nfrom stable_baselines3 import PPO\n\nenv = StockTradingEnv(config, mode="eval", split="train")\nenv.reset(seed=0)\ncols = TICKERS + ["cash"]\n\nfor agent_id in ["price_agent", "sentiment_agent"]:\n    model = PPO.load(f"results/checkpoints/seed{SEED}/{agent_id}_final.zip")\n    weights = []\n    for i in range(env.n_steps):\n        env.start_idx, env.timestep = i, 0\n        a, _ = model.predict(env.observe(agent_id), deterministic=True)\n        a = np.clip(a, 0.0, 1.0)\n        weights.append(a / (a.sum() + 1e-8))\n    w = pd.DataFrame(weights, columns=cols)\n    print(f"\\n{agent_id} — allocation over {len(w)} train days")\n    print(pd.DataFrame({"mean": w.mean(), "std": w.std()}).round(3).T.to_string())'),

    nbf.v4.new_markdown_cell("**Stop here.** Check the policy above before training more seeds. "
                             "For the next seed, set `SEED` in Step 1 and re-run from Step 1."),
]

nb.cells = cells
nb.metadata["colab"] = {"provenance": []}
nb.metadata["kernelspec"] = {"display_name": "Python 3", "name": "python3"}

with open("notebooks/03_training_colab.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Training notebook created at notebooks/03_training_colab.ipynb")
