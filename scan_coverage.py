"""
One pass over the FULL news file. Counts headlines per (ticker, month) for ALL tickers,
so we can see (a) whether the 2020-06 -> 2022 hole is global and (b) which tickers have
continuous coverage. Output is small. Logs everything to scan_coverage.log.

Usage: python scan_coverage.py "<full_news.csv>"
"""
import sys, re
import pandas as pd

src = sys.argv[1]
log = open("scan_coverage.log", "w", encoding="utf-8")
def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); log.write(s + "\n"); log.flush()

def norm(c): return str(c).strip().lower().replace(" ", "_")
cols = [norm(c) for c in pd.read_csv(src, nrows=0).columns]
say("Columns found:", cols)
need = {"date", "stock_symbol"}
if need - set(cols):
    sys.exit(f"Missing columns {need - set(cols)}")

pat = re.compile(r"^\d{4}-\d{2}")
total = bad = 0
parts = []
for chunk in pd.read_csv(src, usecols=lambda c: norm(c) in need, chunksize=500_000,
                         dtype=str, on_bad_lines="skip"):
    chunk.columns = [norm(c) for c in chunk.columns]
    total += len(chunk)
    m = chunk["date"].fillna("").str[:7]
    ok = m.str.match(pat)
    bad += int((~ok).sum())
    g = (pd.DataFrame({"ticker": chunk["stock_symbol"][ok], "month": m[ok]})
         .value_counts().rename("n").reset_index())
    parts.append(g)
    if len(parts) % 5 == 0:
        say(f"scanned {total:,} rows")

df = pd.concat(parts).groupby(["ticker", "month"], as_index=False)["n"].sum()
df.to_csv("monthly_counts.csv", index=False)
say(f"\nTOTAL ROWS SCANNED: {total:,} | rows with unparseable date: {bad:,}")
say(f"Distinct tickers: {df.ticker.nunique():,}")

win = df[(df.month >= "2018-07") & (df.month <= "2023-12")]
say("\n=== ALL tickers: headlines per month (global hole check) ===")
say(win.groupby("month")["n"].sum().to_string())

months = pd.period_range("2018-07", "2023-12", freq="M").astype(str)
piv = win.pivot_table(index="ticker", columns="month", values="n", aggfunc="sum", fill_value=0)
piv = piv.reindex(columns=months, fill_value=0)
cover = (piv >= 5).mean(axis=1) * 100            # % of months with >=5 headlines
tot = piv.sum(axis=1)
res = pd.DataFrame({"headlines": tot, "pct_months_ge5": cover.round(1)})
say("\n=== Tickers with >=90% of months having >=5 headlines (2018-07..2023-12) ===")
good = res[res.pct_months_ge5 >= 90].sort_values("headlines", ascending=False)
say(good.head(40).to_string() if len(good) else "NONE")
say("\nYour five tickers:")
say(res.reindex(["AAPL", "AMZN", "MSFT", "NVDA", "TSLA"]).to_string())
say("\nDone.")
