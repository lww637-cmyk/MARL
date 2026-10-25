# TASK5: implement the price-agent fix (train split only, do not touch test data, do not push)

Decisions (from TASK4 findings): unscaled price features and clip-induced dead gradients caused the collapse.

## 1. Feature standardisation
- Standardise the PRICE observation features (not the sentiment ones, which are already in [-1, 1])
  using the mean and std of the TRAIN split only. Compute once, save to a small JSON (e.g. data/processed/price_scaler.json)
  and commit the code but gitignore nothing needed by Colab: tell me how this file reaches Colab.
- Apply the identical scaling in training, evaluation, the coordinator, SHAP, test_env.py and the Colab notebooks.
- Guard against std = 0. Clip standardised values to a sensible range (e.g. +/-5). Report the min/max after scaling on the train split.

## 2. Action mapping and partner
- In SingleAgentWrapper, normalise the agent's action exactly the way stock_env.step does for evaluation
  (clip to [0,1], divide by the sum, handle sum = 0), so training and evaluation match.
- Replace the random uniform partner with a fixed neutral partner (equal weights across the five stocks and cash). Make this a config
  option (partner: neutral | random) defaulting to neutral, and say where it is documented.

## 3. Tests
Update test_env.py and add tests for the scaler (train-only stats, no test leakage, identical scaling in eval) and the wrapper action mapping.

## 4. Pilot (non-final seeds 98 and 99, 500k steps each, train split only, price agent)
Report per seed: allocation mean and std over the train days (as in the notebook check), average training reward in the first and
last 50k steps, the mean allocation per ticker, and snapshot allocation std every 50k. Compare with the earlier control.
Do not use test data.

## 5. Notebook and report
Update the training notebook (03) so BOTH agents train with the fix. Keep resume support working. Update create_training_notebook.py.
Report files changed, tests, the pilot table, and the manual steps (what to push, what to upload to Drive).
