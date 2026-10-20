import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("""# 03 — MARL Training (Google Colab)

Trains the price agent and the sentiment agent for **one training seed** (`SEED`, default 0)
for `training.total_timesteps` steps each (see `config.yaml`).

Checkpoints are saved every `training.save_frequency` steps to `results/checkpoints/seed{SEED}/`
and copied to `MyDrive/MARL/results/checkpoints_scaled/seed{SEED}/` as soon as each one is written.

**Price-agent fix (TASK5), used by both agents**
- Price features are standardised with the train-split mean/std in `data/price_scaler.json`
  (comes with the git clone) and clipped to +-5, inside the environment, so training,
  evaluation, the coordinator and SHAP all see the same scaled features.
- Each agent's action is mapped to weights exactly as in evaluation (clip to [0, 1], divide by the sum).
- While one agent trains, the other is a fixed partner, `training.partner` in `config.yaml`
  (`neutral` = equal weights, default).
- Checkpoints from before the fix (the old `checkpoints/seed{SEED}/` folder) are never resumed:
  training refuses a folder whose `training_setup.json` does not match the current setup.

**Before running**
1. Push the repo (current `config.yaml`, `src/`, `data/price_scaler.json`).
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

    nbf.v4.new_markdown_cell("## Step 5 — Validate environment (train split)\n"
                             "Also checks the price scaler: scaled train-split price features should have "
                             "mean about 0, std about 1, and stay within +-5."),
    nbf.v4.new_code_cell('import numpy as np\nfrom src.environment.stock_env import StockTradingEnv, PRICE_FEATURES\n\nenv = StockTradingEnv(config, mode="train", split="train")\nprint("Partner while training:", config["training"].get("partner", "neutral"))\nsc = env.price_scaler\nprint(f"Price scaler: {sc[\'n_days\']} train days ({sc[\'train_start\']} .. {sc[\'train_end\']})")\nn_price = len(PRICE_FEATURES) * len(TICKERS)\nobs = []\nfor i in range(env.n_steps):\n    env.start_idx, env.timestep = i, 0\n    obs.append(env.observe("price_agent")[:n_price])\nobs = np.array(obs)\nprint(f"Scaled price features: min {obs.min():.2f} max {obs.max():.2f} "\n      f"mean {obs.mean():.3f} std {obs.std():.3f}")\nassert np.abs(obs).max() <= sc["clip"] and abs(obs.mean()) < 0.05\n\nprint(f"Train split: {env.n_steps} trading days "\n      f"({env.common_dates[0].date()} .. {env.common_dates[-1].date()})")\nprint("Price obs:", env.observation_space("price_agent").shape,\n      "| Sentiment obs:", env.observation_space("sentiment_agent").shape)\n\nfor t in TICKERS:\n    d = env.dataset[t].loc[env.common_dates]\n    print(f"  {t}: has_sentiment on {d.has_sentiment.mean()*100:.1f}% of train days")\n    assert d.has_sentiment.mean() > 0, f"{t} has no sentiment in the train split"'),

    nbf.v4.new_markdown_cell("## Step 6 — Train one seed (resumes automatically)\nEvery checkpoint is copied to Drive as soon as it is saved. "
                             "If Colab disconnects, re-run the notebook from the top: each agent continues from "
                             "its newest checkpoint in `MyDrive/MARL/results/checkpoints_scaled/seed{SEED}/`, "
                             "and an agent with a `_final.zip` there is skipped."),
    nbf.v4.new_code_cell('from src.training.train_marl import find_checkpoints\n\nDRIVE_CKPT = f"{DRIVE}/results/checkpoints_scaled/seed{SEED}"\nTOTAL      = config["training"]["total_timesteps"]\n\nfor agent_id in ["price_agent", "sentiment_agent"]:\n    found = find_checkpoints(agent_id, [DRIVE_CKPT], TOTAL)\n    if found:\n        step, path = found[0]\n        print(f"{agent_id}: will resume from step {step:,} of {TOTAL:,} ({os.path.basename(path)})")\n    else:\n        print(f"{agent_id}: no checkpoint on Drive — starts from step 0")'),
    nbf.v4.new_code_cell('from src.training.train_marl import train\n\nprice_agent, sentiment_agent = train(config, seed=SEED, copy_dir=DRIVE_CKPT)\nprint("Training complete. Checkpoints on Drive:")\nprint(sorted(os.listdir(DRIVE_CKPT)))'),

    nbf.v4.new_markdown_cell("## Step 7 — Policy check (train split only), then STOP\n"
                             "Deterministic weights of each final model on every train-split day "
                             "(same scaled observations and action mapping as training). "
                             "A per-asset std near 0 means the policy ignores its inputs "
                             "(e.g. a constant uniform allocation)."),
    nbf.v4.new_code_cell('from stable_baselines3 import PPO\nfrom src.environment.stock_env import normalise_action\n\nenv = StockTradingEnv(config, mode="eval", split="train")\nenv.reset(seed=0)\ncols = TICKERS + ["cash"]\n\nfor agent_id in ["price_agent", "sentiment_agent"]:\n    model = PPO.load(f"results/checkpoints/seed{SEED}/{agent_id}_final.zip")\n    weights = []\n    for i in range(env.n_steps):\n        env.start_idx, env.timestep = i, 0\n        a, _ = model.predict(env.observe(agent_id), deterministic=True)\n        weights.append(normalise_action(a))\n    w = pd.DataFrame(weights, columns=cols)\n    print(f"\\n{agent_id} — allocation over {len(w)} train days")\n    print(pd.DataFrame({"mean": w.mean(), "std": w.std()}).round(3).T.to_string())'),

    nbf.v4.new_markdown_cell("**Stop here.** Check the policy above before training more seeds. "
                             "For the next seed, set `SEED` in Step 1 and re-run from Step 1."),
]

nb.cells = cells
nb.metadata["colab"] = {"provenance": []}
nb.metadata["kernelspec"] = {"display_name": "Python 3", "name": "python3"}

with open("notebooks/03_training_colab.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Training notebook created at notebooks/03_training_colab.ipynb")
