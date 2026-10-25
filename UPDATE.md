# Update to TASK.md (supersedes conflicting parts of TASK.md)

Agreed on both of your findings. Changes:

1. Split: train = 2018-07-25 to 2022-06-30; test = 2022-07-01 to the end of the data.
2. Evaluation: run ONE fixed full-test-period episode per strategy (same window for
   buy-and-hold, MARL-no-coordinator and MARL+LLM). Keep random-start windows only as a
   secondary robustness option, labelled as overlapping and not independent.
3. Add a training-seed argument to train.py so I can train the agents with several seeds
   (3 to 5) and report variance across training seeds.
4. Step 3: skip the "has_sentiment matches real headline days" check until after
   re-scoring. Only verify (a) ffill(limit=5) with a small synthetic example and
   (b) that train/test dates do not overlap.
5. Prepare a Colab-ready script or notebook cell that runs score_and_save() on
   data/raw/news_headlines.csv with FinBERT on GPU.
6. Wait for the extraction coverage report before editing anything.
