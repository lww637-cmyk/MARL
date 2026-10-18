import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# 03 — MARL Training (Google Colab)\n\nTrains price agent and sentiment agent for 500,000 timesteps each.\n\n**Runtime → Change runtime type → T4 GPU**"),

    nbf.v4.new_markdown_cell("## Step 1 — Check GPU"),
    nbf.v4.new_code_cell('import torch\nprint("CUDA:", torch.cuda.is_available())\nif torch.cuda.is_available():\n    print("Device:", torch.cuda.get_device_name(0))'),

    nbf.v4.new_markdown_cell("## Step 2 — Mount Google Drive"),
    nbf.v4.new_code_cell('from google.colab import drive\ndrive.mount("/content/drive")\nimport os\nos.makedirs("/content/drive/MyDrive/MARL/results/checkpoints", exist_ok=True)\nos.makedirs("/content/drive/MyDrive/MARL/results/logs", exist_ok=True)\nprint("Drive mounted.")'),

    nbf.v4.new_markdown_cell("## Step 3 — Clone repo and install dependencies\nReplace yourusername with your GitHub username."),
    nbf.v4.new_code_cell('!git clone https://github.com/yourusername/MARL.git\n%cd MARL\n!pip install -r requirements.txt -q\nprint("Dependencies installed.")'),

    nbf.v4.new_markdown_cell("## Step 4 — Copy data files from Drive"),
    nbf.v4.new_code_cell('import shutil\n\nos.makedirs("data/raw", exist_ok=True)\nos.makedirs("data/processed", exist_ok=True)\nos.makedirs("data/sentiment", exist_ok=True)\nos.makedirs("results/checkpoints", exist_ok=True)\nos.makedirs("results/logs", exist_ok=True)\n\nimport yaml\nwith open("config.yaml") as f:\n    tickers = yaml.safe_load(f)["data"]["tickers"]\n\nshutil.copy("/content/drive/MyDrive/MARL/data/raw/prices_all.csv",\n            "data/raw/prices_all.csv")\nprint("Copied prices_all.csv")\n\nfor t in tickers:\n    shutil.copy(f"/content/drive/MyDrive/MARL/data/processed/{t}_processed.csv",\n                f"data/processed/{t}_processed.csv")\n    print(f"Copied {t}_processed.csv")\n\nfor t in tickers:\n    src = f"/content/drive/MyDrive/MARL/data/sentiment/{t}_sentiment.csv"\n    if os.path.exists(src):\n        shutil.copy(src, f"data/sentiment/{t}_sentiment.csv")\n        print(f"Copied {t}_sentiment.csv")\n    else:\n        print(f"WARNING: {t}_sentiment.csv not found in Drive")'),

    nbf.v4.new_markdown_cell("## Step 5 — Add .env file with Gemini API key\nReplace YOUR_GEMINI_KEY with your actual key."),
    nbf.v4.new_code_cell('with open(".env", "w") as f:\n    f.write("GEMINI_API_KEY=YOUR_GEMINI_KEY_HERE\\n")\nprint(".env file created.")'),

    nbf.v4.new_markdown_cell("## Step 6 — Validate environment"),
    nbf.v4.new_code_cell('import sys\nsys.path.insert(0, "/content/MARL")\n\nfrom src.training.train_marl import load_config\nfrom src.environment.stock_env import StockTradingEnv\n\nconfig = load_config()\nenv    = StockTradingEnv(config, mode="train")\nprint(f"Environment ready — {env.n_steps} trading days")\nprint(f"Agents: {env.possible_agents}")'),

    nbf.v4.new_markdown_cell("## Step 7 — Run full training\nExpect ~60-90 minutes on T4 GPU for 500,000 timesteps per agent."),
    nbf.v4.new_code_cell('from src.training.train_marl import train\nprice_agent, sentiment_agent = train(config)\nprint("Training complete.")'),

    nbf.v4.new_markdown_cell("## Step 8 — Copy checkpoints to Drive"),
    nbf.v4.new_code_cell('files_to_save = [\n    "results/checkpoints/price_agent_final.zip",\n    "results/checkpoints/sentiment_agent_final.zip",\n]\n\nfor f in files_to_save:\n    if os.path.exists(f):\n        dst = f"/content/drive/MyDrive/MARL/{f}"\n        os.makedirs(os.path.dirname(dst), exist_ok=True)\n        shutil.copy(f, dst)\n        print(f"Saved {f} to Drive")\n    else:\n        print(f"WARNING: {f} not found")'),

    nbf.v4.new_markdown_cell("## Step 9 — Run baselines"),
    nbf.v4.new_code_cell('from src.training.train_baseline import run_all_baselines\n\nsummary = run_all_baselines(\n    config,\n    price_ckpt     = "results/checkpoints/price_agent_final.zip",\n    sentiment_ckpt = "results/checkpoints/sentiment_agent_final.zip"\n)\nprint(summary)'),

    nbf.v4.new_markdown_cell("## Step 10 — Save all results to Drive"),
    nbf.v4.new_code_cell('result_files = [\n    "results/logs/baseline_results.csv",\n    "results/logs/baseline_summary.csv",\n]\n\nfor f in result_files:\n    if os.path.exists(f):\n        dst = f"/content/drive/MyDrive/MARL/{f}"\n        os.makedirs(os.path.dirname(dst), exist_ok=True)\n        shutil.copy(f, dst)\n        print(f"Saved {f} to Drive")\n\nprint("All results saved to Google Drive.")\nprint("Download and place in your local results/ folder.")'),
]

nb.cells = cells

with open("notebooks/03_training_colab.ipynb", "w", encoding="utf-8") as f:
    import nbformat
    f.write(nbformat.writes(nb))

print("Training notebook created at notebooks/03_training_colab.ipynb")