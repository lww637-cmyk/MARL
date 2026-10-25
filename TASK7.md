Create policy_check.py that does exactly what Step 7 (policy check) does in
notebooks/03_training_colab.ipynb, but locally. Arguments: --train-seed N
(default 0). It loads the final price and sentiment checkpoints for that seed
from results/checkpoints_scaled/seed{N} using load_checked_agents() (the setup
check must stay on), runs both agents on the TRAINING split only, and prints,
for each agent, the same table as Step 7: per-ticker allocation mean and std,
plus cash. Do not touch the test period, do not call Gemini, and do not change
any training code, the scaler, or mlflow.db. Then run it for --train-seed 4 and
--train-seed 5 and show me both outputs.