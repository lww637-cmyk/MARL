# TASK6: checkpoint safety check and path unification (do not push, do not train, do not call Gemini)

## 1. Safety check
Evaluation, SHAP and coordinator code (evaluate_agents.py, compute_shap.py, src/coordinator, anything else that
loads agents) must, before loading a checkpoint folder, read its training_setup.json and compare it with the
current setup (price scaler hash from data/price_scaler.json, training.partner, action mapping).
If the file is missing or any field differs, stop with a clear error naming the folder and the mismatch.
Reuse the check that train_marl.py already uses for resume. Make it one shared function.

## 2. Paths
Make the checkpoint root a single config setting (e.g. paths.checkpoints_root, default results/checkpoints_scaled)
and make seeds 0, 1 and 2 load from it as {root}/seed{N}/price_agent_final.zip and sentiment_agent_final.zip.
Tell me where seeds 1 and 2 were actually written locally and give me the exact commands to move them if needed.
Make evaluate_agents.py accept --checkpoint-root and --train-seeds.

## 3. Tests
Add tests: an old folder without training_setup.json is refused; a folder with a wrong hash is refused;
a matching folder passes. Run the existing tests.

Report files changed, test results and the move commands. Do not push.
