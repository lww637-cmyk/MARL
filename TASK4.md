# TASK4: diagnose the price-agent collapse (train split only; do not touch test data; do not push)

Observation: seed 0 price agent outputs a constant allocation on all 958 train days
(AMD 0.333, GS 0.333, CVX 0.333, others 0.0, std 0.000). Sentiment agent is healthy
(allocation std 0.15 to 0.25). Earlier the price agent collapsed to uniform 1/6 on the old tickers.

## A. Read and report (no running yet)
1. How is the agent's action turned into portfolio weights (clip, softmax, abs/normalise)?
   What are the action-space bounds, and is the Gaussian policy mean unbounded?
2. PPO settings in base_agent.py for the price agent (ent_coef, n_steps, batch, lr, policy_kwargs,
   net_arch, any use_sde or squash). Is VecNormalize used for observations or rewards?
3. Price observation: min, max, mean, std per column group over the train split. Are any
   features on very different scales (raw prices, MACD, volume)? Is the reward scale large?
4. Does sentiment-agent training depend on the price agent (e.g. as a fixed partner), or are
   the two trained independently? State exactly how the other agent acts during each one's training.

## B. Checkpoint analysis (seed 0 price-agent checkpoints from Drive, 100k to 2M)
For each checkpoint print mean and std of allocation over the train split, the policy log_std,
and the mean absolute action before clipping. Tell me at which step it collapses
(from the start, or later).

## C. Pilot fixes (200k steps each, train split only, seed 99, which is not a final seed)
Try and compare, one change at a time, on allocation std and mean train reward:
 1. observation normalisation (VecNormalize obs only, saving stats for evaluation)
 2. action mapping via softmax over unbounded logits instead of clip-and-normalise
 3. both 1 and 2
 4. the current setup as a control
Do not tune on test data. Report a table. Do not change defaults; propose only.

## D. Report
Findings for A, the collapse step from B, the table from C, and your ranked recommendation.
