"""
Extract headlines for the project tickers from the FULL news dataset (e.g. FNSPID).
Streams the file in chunks, keeps only date / headline / ticker, does NOT require
article text, de-duplicates, and reports coverage gaps.

Usage:  python extract_news.py <full_news.csv> [output.csv] [TICK1,TICK2,...]
"""
import sys
import numpy as np
import pandas as pd

TICKERS = (sys.argv[3].split(",") if len(sys.argv) > 3 else ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA"])
START, END = "2018-07-25", "2023-12-28"

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "data/raw/news_headlines.csv"

def norm(c):
    return str(c).strip().lower().replace(" ", "_")

header = [norm(c) for c in pd.read_csv(src, nrows=0).columns]
need = {"date", "article_title", "stock_symbol"}
missing = need - set(header)
if missing:
    sys.exit(f"Missing expected columns {missing}. Found: {header}")

kept, total = [], 0
for chunk in pd.read_csv(src, usecols=lambda c: norm(c) in need, chunksize=200_000,
                         on_bad_lines="error"):
    chunk.columns = [norm(c) for c in chunk.columns]
    total += len(chunk)
    chunk = chunk[chunk["stock_symbol"].isin(TICKERS)]
    if chunk.empty:
        continue
    chunk = chunk.copy()
    chunk["date"] = pd.to_datetime(chunk["date"], utc=True, errors="coerce")
    chunk = chunk.dropna(subset=["date", "article_title"])
    chunk["date"] = chunk["date"].dt.tz_localize(None).dt.normalize()
    chunk = chunk[(chunk["date"] >= START) & (chunk["date"] <= END)]
    kept.append(chunk)
    print(f"  scanned {total:,} rows | kept {sum(len(k) for k in kept):,}", end="\r")

df = pd.concat(kept, ignore_index=True)
before = len(df)
df = df.drop_duplicates(subset=["date", "article_title", "stock_symbol"])
df = df.sort_values(["stock_symbol", "date"]).reset_index(drop=True)
print(f"\nScanned {total:,} rows -> kept {len(df):,} headlines ({before-len(df):,} duplicates removed)")

df.to_csv(out, index=False)
print("Saved ->", out)

# ---- coverage report (gaps include the stretch before first / after last headline) ----
print("\nHeadlines per ticker:")
print(df["stock_symbol"].value_counts().to_string())

bdays = pd.bdate_range(START, END)
rows = []
for t in TICKERS:
    days = pd.DatetimeIndex(np.sort(df.loc[df.stock_symbol == t, "date"].unique()))
    if len(days) == 0:
        rows.append((t, 0, 0.0, None, None, 9999)); continue
    edges = pd.DatetimeIndex([pd.Timestamp(START)]).append(days).append(pd.DatetimeIndex([pd.Timestamp(END)]))
    gap = int(np.diff(edges.values).astype("timedelta64[D]").astype(int).max())
    rows.append((t, len(df[df.stock_symbol == t]), round(bdays.isin(days).mean() * 100, 1),
                 days[0].date(), days[-1].date(), gap))
rep = pd.DataFrame(rows, columns=["ticker", "headlines", "pct_bdays_covered", "first", "last", "longest_gap_days"])
rep = rep.sort_values(["longest_gap_days", "pct_bdays_covered"], ascending=[True, False])
print("\nCoverage ranked by longest gap (calendar days, edges included), best first:")
print(rep.to_string(index=False))
rep.to_csv(out.replace(".csv", "_coverage.csv"), index=False)
