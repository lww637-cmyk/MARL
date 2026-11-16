# Master table: test period 2022-07-01 to 2023-12-15

Mean ± std over training seeds (std with ddof=1; none for single deterministic runs). Max drawdown is the true peak-to-trough drawdown (max_drawdown_true). MARL+LLM: replayed runs in results/logs/replay_final3, coordinator interval N=20. Built by make_master_table.py.

| Strategy | n | Total return (%) | Sharpe | Max drawdown (%) | Turnover | Total cost ($) |
|---|---:|---:|---:|---:|---:|---:|
| buy_and_hold | 1 | 37.5 | 1.05 | 19.9 | 2.00 | 200 |
| equal_weight_daily_rebalanced | 1 | 41.7 | 1.13 | 20.0 | 2.00 | 200 |
| MARL-no-coordinator | 6 | 4.5 ± 13.4 | 0.25 ± 0.29 | 36.9 ± 6.3 | 245.11 ± 42.60 | 23,552 ± 4,887 |
| price-agent-only | 6 | -5.2 ± 10.0 | 0.02 ± 0.22 | 36.0 ± 8.1 | 253.77 ± 39.12 | 23,670 ± 5,318 |
| sentiment-agent-only | 6 | 22.5 ± 34.4 | 0.45 ± 0.43 | 43.1 ± 5.5 | 221.60 ± 123.97 | 20,579 ± 10,550 |
| price-only-hold20 | 6 | 34.0 ± 19.1 | 0.83 ± 0.34 | 31.4 ± 7.7 | 25.75 ± 2.91 | 3,005 ± 428 |
| sentiment-only-hold20 | 6 | 63.8 ± 40.6 | 0.93 ± 0.38 | 36.6 ± 6.3 | 13.34 ± 7.66 | 1,418 ± 764 |
| MARL-average-hold20 | 6 | 41.9 ± 13.0 | 0.92 ± 0.16 | 33.2 ± 8.3 | 17.75 ± 0.75 | 2,014 ± 143 |
| MARL-average-hold20-cash3 | 6 | 41.6 ± 14.1 | 0.93 ± 0.19 | 32.3 ± 8.1 | 17.49 ± 1.20 | 1,984 ± 164 |
| MARL-average-hold20-cash5.5 | 6 | 39.3 ± 15.6 | 0.90 ± 0.23 | 32.3 ± 6.8 | 17.13 ± 1.25 | 1,923 ± 184 |
| MARL-average-hold20-cash8 | 6 | 35.5 ± 12.5 | 0.86 ± 0.18 | 31.5 ± 6.7 | 16.65 ± 1.11 | 1,846 ± 147 |
| price-static-hold20 | 6 | 44.0 ± 5.4 | 1.18 ± 0.02 | 21.7 ± 2.7 | 1.74 ± 0.07 | 174 ± 7 |
| sentiment-static-hold20 | 6 | 47.1 ± 2.8 | 1.19 ± 0.02 | 23.4 ± 2.2 | 1.71 ± 0.23 | 171 ± 23 |
| MARL+LLM | 6 | 45.2 ± 14.7 | 1.09 ± 0.23 | 24.2 ± 1.8 | 16.93 ± 0.95 | 1,966 ± 190 |

## LaTeX

```latex
\begin{tabular}{l r r r r r r }
\toprule
Strategy & n & Total return (\%) & Sharpe & Max drawdown (\%) & Turnover & Total cost (\$) \\
\midrule
buy\_and\_hold & 1 & 37.5 & 1.05 & 19.9 & 2.00 & 200 \\
equal\_weight\_daily\_rebalanced & 1 & 41.7 & 1.13 & 20.0 & 2.00 & 200 \\
MARL-no-coordinator & 6 & 4.5 $\pm$ 13.4 & 0.25 $\pm$ 0.29 & 36.9 $\pm$ 6.3 & 245.11 $\pm$ 42.60 & 23,552 $\pm$ 4,887 \\
price-agent-only & 6 & -5.2 $\pm$ 10.0 & 0.02 $\pm$ 0.22 & 36.0 $\pm$ 8.1 & 253.77 $\pm$ 39.12 & 23,670 $\pm$ 5,318 \\
sentiment-agent-only & 6 & 22.5 $\pm$ 34.4 & 0.45 $\pm$ 0.43 & 43.1 $\pm$ 5.5 & 221.60 $\pm$ 123.97 & 20,579 $\pm$ 10,550 \\
price-only-hold20 & 6 & 34.0 $\pm$ 19.1 & 0.83 $\pm$ 0.34 & 31.4 $\pm$ 7.7 & 25.75 $\pm$ 2.91 & 3,005 $\pm$ 428 \\
sentiment-only-hold20 & 6 & 63.8 $\pm$ 40.6 & 0.93 $\pm$ 0.38 & 36.6 $\pm$ 6.3 & 13.34 $\pm$ 7.66 & 1,418 $\pm$ 764 \\
MARL-average-hold20 & 6 & 41.9 $\pm$ 13.0 & 0.92 $\pm$ 0.16 & 33.2 $\pm$ 8.3 & 17.75 $\pm$ 0.75 & 2,014 $\pm$ 143 \\
MARL-average-hold20-cash3 & 6 & 41.6 $\pm$ 14.1 & 0.93 $\pm$ 0.19 & 32.3 $\pm$ 8.1 & 17.49 $\pm$ 1.20 & 1,984 $\pm$ 164 \\
MARL-average-hold20-cash5.5 & 6 & 39.3 $\pm$ 15.6 & 0.90 $\pm$ 0.23 & 32.3 $\pm$ 6.8 & 17.13 $\pm$ 1.25 & 1,923 $\pm$ 184 \\
MARL-average-hold20-cash8 & 6 & 35.5 $\pm$ 12.5 & 0.86 $\pm$ 0.18 & 31.5 $\pm$ 6.7 & 16.65 $\pm$ 1.11 & 1,846 $\pm$ 147 \\
price-static-hold20 & 6 & 44.0 $\pm$ 5.4 & 1.18 $\pm$ 0.02 & 21.7 $\pm$ 2.7 & 1.74 $\pm$ 0.07 & 174 $\pm$ 7 \\
sentiment-static-hold20 & 6 & 47.1 $\pm$ 2.8 & 1.19 $\pm$ 0.02 & 23.4 $\pm$ 2.2 & 1.71 $\pm$ 0.23 & 171 $\pm$ 23 \\
MARL+LLM & 6 & 45.2 $\pm$ 14.7 & 1.09 $\pm$ 0.23 & 24.2 $\pm$ 1.8 & 16.93 $\pm$ 0.95 & 1,966 $\pm$ 190 \\
\bottomrule
\end{tabular}
```
