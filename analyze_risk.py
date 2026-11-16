"""Risk by volatility regime, true drawdown depth and recovery time (test period).

Reads logged daily equity curves only (no Gemini, no environment runs):
  results/logs/dd_test/equity_curves.csv          11 strategies, N=20 controls
  results/logs/replay_final3/equity_curves.csv     MARL+LLM, N=20 (headline run)
  results/logs/replay_final3_repeat/equity_curves.csv  MARL+LLM, N=20 repeat
  results/logs/replay_llm_cf10/equity_curves.csv   MARL+LLM, N=10
  results/logs/replay_llm_cf40/equity_curves.csv   MARL+LLM, N=40
  results/logs/dd_train/equity_curves.csv          training-period
                                                   equal_weight_daily_rebalanced,
                                                   used ONLY to fix the threshold

Regimes (pre-specified, fixed from training data only)
  r_t  = daily return of equal_weight_daily_rebalanced (V_t / V_{t-1} - 1).
  vol_t = sample std (ddof=1) of the 20 returns r_{t-20} .. r_{t-1}, i.e. the
         trailing 20-day realised volatility known BEFORE day t (so a large
         move on day t cannot classify its own day). Unannualised.
  threshold = 75th percentile of vol_t over the training-split days that have
         a full 20-return window. Nothing from the test period enters it.
  A test day is "volatile" if vol_t > threshold, otherwise "calm".
  The train and test equal-weight returns are concatenated so the first test
  days have a full window; the one return between the last training close
  (2022-06-30) and the test start value (2022-07-01) is not logged and is
  simply absent from the window (windows spanning it use the 20 adjacent
  logged returns).
  The same regime calendar applies to every strategy and seed, so all
  strategies are compared on exactly the same days.

Per strategy, seed and regime (calm / volatile / all):
  n_days, share_days (of the 367 test returns), mean_daily_return,
  sharpe_ann = mean / std(ddof=0) * sqrt(252) (as StockTradingEnv, no
  risk-free rate), worst_day (min daily return), es95 = mean of the daily
  returns at or below the empirical 5% quantile (a return, negative = loss).
  With ~90 volatile days the volatile-regime ES rests on ~5 observations.

Drawdown (whole test path, per strategy and seed):
  max_drawdown_true = max_t (running peak - V_t) / running peak, the same
  definition as evaluate_agents.true_max_drawdown; checked against the logged
  test_period_results.csv value.
  peak_date / trough_date of that deepest drawdown; decline_days = trading
  days peak -> trough; recovery_days = trading days trough -> first close at
  or above the old peak; recovered = False if that never happens before the
  end of the test (then recovery_days is empty and underwater_days_at_end
  gives the censored trough -> end count). max_underwater_days = longest
  spell (trading days) spent below a previous peak.

Writes (existing results are not touched):
  report/tables/risk_regimes.csv            long: strategy x seed x regime
  report/tables/risk_regimes_by_strategy.csv mean / std over seeds
  report/tables/risk_summary.csv            drawdown depth and recovery per seed
  report/tables/risk_regime_calendar.csv    date, vol_t, regime (test days)
  report/tables/risk_drawdown_series.csv    daily drawdown series for plots
  report/tables/risk_regime_meta.json       threshold and definitions
"""
import json
import os

import numpy as np
import pandas as pd

LOGS = "results/logs"
OUT_DIR = "report/tables"
WINDOW = 20
QUANTILE = 0.75
ES_LEVEL = 0.05
EQUAL_WEIGHT = "equal_weight_daily_rebalanced"

# folder -> (label suffix for MARL+LLM, coordinator interval N)
SOURCES = {
    "dd_test": (None, 20),
    "replay_final3": ("", 20),
    "replay_final3_repeat": ("-repeat", 20),
    "replay_llm_cf10": ("-N10", 10),
    "replay_llm_cf40": ("-N40", 40),
}


def load_curve(folder: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(LOGS, folder, "equity_curves.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df["source"] = folder
    return df


def ew_returns(df: pd.DataFrame) -> pd.Series:
    ew = df[df["strategy"] == EQUAL_WEIGHT].sort_values("step")
    v = ew.set_index("date")["portfolio_value"]
    return v.pct_change().dropna()


def trailing_vol(r: pd.Series) -> pd.Series:
    """Std of the WINDOW returns strictly before each day."""
    return r.rolling(WINDOW, min_periods=WINDOW).std(ddof=1).shift(1)


def es(x: np.ndarray, level: float = ES_LEVEL) -> float:
    if len(x) == 0:
        return np.nan
    q = np.quantile(x, level)
    return float(x[x <= q].mean())


def sharpe(x: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0:
        return np.nan
    return float(np.mean(x) / np.std(x) * np.sqrt(252))


def regime_stats(r: np.ndarray, n_total: int) -> dict:
    return {
        "n_days": len(r),
        "share_days": len(r) / n_total,
        "mean_daily_return": float(np.mean(r)) if len(r) else np.nan,
        "sharpe_ann": sharpe(r),
        "worst_day": float(np.min(r)) if len(r) else np.nan,
        "es95": es(r),
    }


def drawdown_stats(dates: pd.Series, v: np.ndarray) -> dict:
    peak = np.maximum.accumulate(v)
    dd = (peak - v) / peak
    trough = int(np.argmax(dd))
    depth = float(dd[trough])
    out = {"max_drawdown_true": depth}
    if depth == 0:
        out.update(peak_date=None, trough_date=None, recovery_date=None,
                   decline_days=0, recovery_days=0, recovered=True,
                   underwater_days_at_end=0)
    else:
        p = int(np.where(v[:trough + 1] == peak[trough])[0][-1])
        after = np.where(v[trough:] >= peak[trough])[0]
        rec = trough + int(after[0]) if len(after) else None
        out.update(
            peak_date=str(dates.iloc[p].date()),
            trough_date=str(dates.iloc[trough].date()),
            recovery_date=str(dates.iloc[rec].date()) if rec is not None else None,
            decline_days=trough - p,
            recovery_days=(rec - trough) if rec is not None else np.nan,
            recovered=rec is not None,
            underwater_days_at_end=(len(v) - 1 - trough) if rec is None else 0,
        )
    # longest spell below a previous peak (censored at the end of the test)
    longest = run = 0
    for under in dd > 0:
        run = run + 1 if under else 0
        longest = max(longest, run)
    out["max_underwater_days"] = longest
    out["final_drawdown"] = float(dd[-1])
    return out, peak, dd


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── regime calendar ────────────────────────────────────────────────────
    train = load_curve("dd_train")
    test_controls = load_curve("dd_test")
    r_train = ew_returns(train)
    r_test = ew_returns(test_controls)
    assert r_train.index.max() < r_test.index.min()
    vol = trailing_vol(pd.concat([r_train, r_test]))
    vol_train = vol.loc[r_train.index].dropna()
    threshold = float(vol_train.quantile(QUANTILE))
    vol_test = vol.loc[r_test.index]
    assert vol_test.notna().all()
    regime = pd.Series(np.where(vol_test > threshold, "volatile", "calm"),
                       index=vol_test.index)
    calendar = pd.DataFrame({"date": regime.index.date,
                             "ew_return": r_test.values,
                             "trailing_vol_20d": vol_test.values,
                             "trailing_vol_20d_ann": vol_test.values * np.sqrt(252),
                             "regime": regime.values})
    calendar.to_csv(os.path.join(OUT_DIR, "risk_regime_calendar.csv"), index=False)
    n_vol = int((regime == "volatile").sum())
    print(f"Threshold (train 75th pct of trailing 20d vol): {threshold:.5f} "
          f"daily = {threshold*np.sqrt(252):.3f} annualised "
          f"(from {len(vol_train)} training days)")
    print(f"Test days: {len(regime)}  volatile {n_vol} "
          f"({n_vol/len(regime):.1%})  calm {len(regime)-n_vol}")

    # ── per strategy / seed ────────────────────────────────────────────────
    logged = pd.read_csv(os.path.join(LOGS, "dd_test", "test_period_results.csv"))
    regime_rows, summary_rows, series = [], [], []
    for folder, (suffix, n) in SOURCES.items():
        df = test_controls if folder == "dd_test" else load_curve(folder)
        if suffix is not None:
            df = df.assign(strategy="MARL+LLM" + suffix)
        for (strategy, seed), g in df.groupby(["strategy", "train_seed"],
                                              dropna=False, sort=False):
            g = g.sort_values("step")
            v = g["portfolio_value"].to_numpy(dtype=np.float64)
            dates = g["date"].reset_index(drop=True)
            r = pd.Series(v[1:] / v[:-1] - 1, index=dates.iloc[1:].values)
            assert r.index.equals(regime.index), (folder, strategy, seed)
            seed_val = None if pd.isna(seed) else int(seed)
            key = {"strategy": strategy, "train_seed": seed_val,
                   "interval_N": n, "source": folder}

            for name, mask in [("calm", regime == "calm"),
                               ("volatile", regime == "volatile"),
                               ("all", pd.Series(True, index=regime.index))]:
                regime_rows.append({**key, "regime": name,
                                    **regime_stats(r[mask.values].to_numpy(),
                                                   len(r))})

            dd_info, peak, dd = drawdown_stats(dates, v)
            if folder == "dd_test":
                sel = logged[logged["strategy"] == strategy]
                sel = (sel[sel["train_seed"].isna()] if seed_val is None
                       else sel[sel["train_seed"] == seed_val])
                assert len(sel) == 1
                assert np.isclose(sel["max_drawdown_true"].iloc[0],
                                  dd_info["max_drawdown_true"], atol=1e-9)
            summary_rows.append({**key, "total_return": v[-1] / v[0] - 1,
                                 **dd_info})
            series.append(pd.DataFrame({
                **{k: [val] * len(v) for k, val in key.items()},
                "step": g["step"].to_numpy(), "date": dates.dt.date,
                "portfolio_value": v, "running_peak": peak, "drawdown": dd,
                "regime": ["start"] + list(regime.values)}))

    reg = pd.DataFrame(regime_rows)
    reg.to_csv(os.path.join(OUT_DIR, "risk_regimes.csv"), index=False)
    summ = pd.DataFrame(summary_rows)
    summ.to_csv(os.path.join(OUT_DIR, "risk_summary.csv"), index=False)
    pd.concat(series, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "risk_drawdown_series.csv"), index=False)

    metrics = ["mean_daily_return", "sharpe_ann", "worst_day", "es95"]
    agg = (reg.groupby(["strategy", "interval_N", "source", "regime"], sort=False)
           .agg(n_seeds=("sharpe_ann", "size"), share_days=("share_days", "first"),
                **{f"{m}_{s}": (m, s) for m in metrics for s in ("mean", "std")})
           .reset_index())
    agg.to_csv(os.path.join(OUT_DIR, "risk_regimes_by_strategy.csv"), index=False)

    meta = {
        "threshold_daily_vol": threshold,
        "threshold_annualised_vol": threshold * np.sqrt(252),
        "threshold_quantile": QUANTILE, "window": WINDOW,
        "vol_definition": "sample std (ddof=1) of the 20 equal-weight daily "
                          "returns before day t (excludes day t)",
        "threshold_source": "results/logs/dd_train equal_weight_daily_rebalanced",
        "train_days_used": int(len(vol_train)),
        "train_first_last": [str(vol_train.index.min().date()),
                             str(vol_train.index.max().date())],
        "test_days": int(len(regime)), "volatile_days": n_vol,
        "sharpe": "mean/std(ddof=0)*sqrt(252), no risk-free rate",
        "es95": "mean of daily returns <= empirical 5% quantile",
        "sources": list(SOURCES),
    }
    with open(os.path.join(OUT_DIR, "risk_regime_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    pd.set_option("display.width", 200)
    show = agg[agg["source"].isin(["dd_test", "replay_final3"])]
    print(show.pivot_table(index="strategy", columns="regime",
                           values=["sharpe_ann_mean", "es95_mean"]).round(3))
    print(summ.groupby("strategy", sort=False)[
        ["max_drawdown_true", "recovery_days", "recovered", "max_underwater_days"]]
        .agg({"max_drawdown_true": "mean", "recovery_days": "mean",
              "recovered": "sum", "max_underwater_days": "mean"}).round(3))


if __name__ == "__main__":
    main()
