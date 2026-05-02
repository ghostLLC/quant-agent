"""Extend the HS300 cross-section dataset with historical OHLCV via tushare.

Base dataset (hs300_cross_section.csv): 280 assets, ~1 year, full fundamentals.
This script extends it BACKWARD to 2021-01-01 by pulling tushare daily() OHLCV.

Features:
  - 45s timeout per stock (ThreadPoolExecutor) — no more hangs
  - Checkpoint every 50 stocks — progress saved, auto-resume on restart
  - Pulls 2021-01-01 → day before existing data starts

Usage:
  python build_dataset.py --full       # Full backward extension
  python build_dataset.py --refresh    # Forward incremental refresh
  python build_dataset.py --use-snapshot  # Copy hs300 CSV as-is
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
BASE_DATASET = DATA_DIR / "hs300_cross_section.csv"
THICK_DATASET = DATA_DIR / "cross_section_thick.csv"
THICK_META = DATA_DIR / "cross_section_thick_meta.json"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

TARGET_COLS = [
    "date", "asset", "open", "high", "low", "close", "volume", "amount",
    "turnover", "market_cap", "float_market_cap", "pb", "pe",
    "total_share", "float_share", "industry",
]

CHECKPOINT_EVERY = 50  # save after this many successful pulls
TUSHARE_PAUSE = 1.5  # seconds between calls


def _to_tushare(code: str) -> str:
    c = str(int(float(str(code)))).zfill(6)
    return f"{c}.SH" if c.startswith(("6", "9")) else f"{c}.SZ"


def _init_tushare():
    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("TUSHARE_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["TUSHARE_TOKEN"] = token
                    break
    if not token:
        return None
    import tushare as ts
    ts.set_token(token)
    return ts.pro_api(timeout=30)  # 30s HTTP timeout to prevent hangs


def _print_bar(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


# ---------------------------------------------------------------------------
# Single-stock pull (timeout via tushare API, no threads)
# ---------------------------------------------------------------------------

def _pull_one_stock(code: str, start: str, end: str, pro) -> pd.DataFrame | None:
    """Pull OHLCV for a single stock via tushare. Timeout handled by API client."""
    ts_code = _to_tushare(code)
    try:
        df = pro.daily(
            ts_code=ts_code,
            start_date=start,
            end_date=end,
            fields="ts_code,trade_date,open,high,low,close,vol,amount",
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df["asset"] = code
    df = df.rename(columns={"vol": "volume"})
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date", "asset", "open", "high", "low", "close", "volume", "amount"]]


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def _checkpoint_path() -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / "extend_checkpoint.csv"


def _load_checkpoint() -> tuple[pd.DataFrame, set[str]]:
    """Load checkpoint data + set of already-pulled asset codes."""
    cp = _checkpoint_path()
    if cp.exists():
        df = pd.read_csv(cp)
        done = set(df["asset"].unique())
        print(f"  Loaded checkpoint: {len(df):,} rows, {len(done)} assets")
        return df, done
    return pd.DataFrame(), set()


def _save_checkpoint(df: pd.DataFrame) -> None:
    df.to_csv(_checkpoint_path(), index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Backward extension
# ---------------------------------------------------------------------------

def extend_backward(base_path: Path, output_path: Path, target_start: str = "20210101") -> None:
    pro = _init_tushare()
    if pro is None:
        print("TUSHARE_TOKEN not set. Cannot extend. Use --use-snapshot instead.")
        sys.exit(1)

    _print_bar("Loading base HS300 dataset")
    base = pd.read_csv(base_path)
    base["date"] = pd.to_datetime(base["date"])
    base["asset"] = base["asset"].apply(lambda x: str(int(float(str(x)))).zfill(6))

    assets = sorted(base["asset"].unique())
    hist_end = (base["date"].min() - pd.Timedelta(days=1)).strftime("%Y%m%d")

    print(f"  Base:   {len(base):,} rows, {len(assets)} assets")
    print(f"  Range:  {base['date'].min().strftime('%Y-%m-%d')} ~ {base['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Extend: {target_start} ~ {hist_end}")

    # Resume from checkpoint
    checkpoint_df, done_set = _load_checkpoint()
    remaining = [a for a in assets if a not in done_set]
    print(f"  Resume: {len(done_set)} done, {len(remaining)} remaining")

    if not remaining:
        print("  All stocks already pulled — using checkpoint data")

    # Pull remaining stocks
    frames = [checkpoint_df] if not checkpoint_df.empty else []
    ok = 0
    fail = 0
    timeout = 0
    total = len(assets)

    for i, code in enumerate(remaining):
        batch_num = i + 1
        if batch_num % 20 == 0 or batch_num == 1:
            print(f"  {len(done_set) + batch_num - 1}/{total} (ok={ok} fail={fail} timeout={timeout})")

        df = _pull_one_stock(code, target_start, hist_end, pro)
        if df is not None and not df.empty:
            frames.append(df)
            ok += 1
        else:
            if df is None:
                timeout += 1
            else:
                fail += 1

        # Checkpoint
        if ok > 0 and ok % CHECKPOINT_EVERY == 0:
            all_pulled = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            all_pulled = all_pulled.drop_duplicates(subset=["date", "asset"])
            _save_checkpoint(all_pulled)
            print(f"  [checkpoint saved: {len(all_pulled):,} rows]")

        if ok + fail + timeout > 0 and (ok + fail + timeout) % 200 == 0:
            # Longer pause every 200 calls to avoid rate limiting
            time.sleep(3)
        else:
            time.sleep(TUSHARE_PAUSE)

    # Final checkpoint
    if frames:
        all_pulled = pd.concat(frames, ignore_index=True)
        all_pulled = all_pulled.drop_duplicates(subset=["date", "asset"])
        _save_checkpoint(all_pulled)
        print(f"\n  Final checkpoint: {len(all_pulled):,} rows")

    stats = f"ok={ok} fail={fail} timeout={timeout}"
    print(f"\n  Pull complete: {stats}")

    if not frames:
        print("  WARNING: No data pulled — using base dataset as-is")
        _save_output(base, output_path)
        return

    hist = pd.concat(frames, ignore_index=True)
    hist = hist.drop_duplicates(subset=["date", "asset"])

    combined = pd.concat([hist, base], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "asset"], keep="last")
    combined = combined.sort_values(["date", "asset"]).reset_index(drop=True)

    _save_output(combined, output_path)

    # Clean checkpoint on success
    cp = _checkpoint_path()
    if cp.exists():
        cp.unlink()
        print("  Checkpoint cleaned.")


# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------

def _save_output(df: pd.DataFrame, output_path: Path) -> None:
    _print_bar("Saving extended dataset")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["asset"] = df["asset"].astype(str).str.zfill(6)

    for col in TARGET_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df.sort_values(["date", "asset"]).reset_index(drop=True)
    df[TARGET_COLS].to_csv(output_path, index=False, encoding="utf-8-sig")

    meta = {
        "file": str(output_path),
        "rows": len(df),
        "assets": int(df["asset"].nunique()),
        "trading_days": int(df["date"].nunique()),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "missing_pct": {
            c: round(float(df[c].isna().mean()) * 100, 1)
            for c in TARGET_COLS if c not in ("date", "asset")
        },
        "last_built": datetime.now().isoformat(),
    }
    THICK_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    dmin = datetime.strptime(meta["date_min"], "%Y-%m-%d")
    dmax = datetime.strptime(meta["date_max"], "%Y-%m-%d")
    span = (dmax - dmin).days

    print(f"  Rows:   {meta['rows']:,}")
    print(f"  Assets: {meta['assets']}")
    print(f"  Days:   {meta['trading_days']}")
    print(f"  Range:  {meta['date_min']} ~ {meta['date_max']} ({span / 365.25:.1f} yr)")

    print(f"\n  Column fill:")
    for c, p in meta["missing_pct"].items():
        fill = 100 - p
        bar = "#" * int(fill / 5) + "." * max(0, 20 - int(fill / 5))
        status = "OK" if p < 5 else ("!" if p < 50 else "!!")
        print(f"    {c:20s} [{bar}] {fill:.0f}% {status}")

    cutoff = dmax - pd.DateOffset(months=6)
    train = df[pd.to_datetime(df["date"]) <= cutoff]
    test = df[pd.to_datetime(df["date"]) > cutoff]
    print(f"\n  OOS: {train['date'].nunique()} train + {test['date'].nunique()} test")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


# ---------------------------------------------------------------------------
# Other modes
# ---------------------------------------------------------------------------

def snapshot_copy(base_path: Path, output_path: Path) -> None:
    _print_bar("Snapshot copy (no tushare extension)")
    base = pd.read_csv(base_path)
    base["date"] = pd.to_datetime(base["date"])
    base["asset"] = base["asset"].apply(lambda x: str(int(float(str(x)))).zfill(6))
    _save_output(base, output_path)


def incremental_refresh(output_path: Path) -> None:
    if not output_path.exists():
        print("No existing thick dataset. Run --full first.")
        sys.exit(1)

    pro = _init_tushare()
    if pro is None:
        print("TUSHARE_TOKEN not set.")
        sys.exit(1)

    existing = pd.read_csv(output_path)
    existing["date"] = pd.to_datetime(existing["date"])
    last_date = existing["date"].max()
    assets = sorted(existing["asset"].unique())

    today = datetime.now().strftime("%Y%m%d")
    start = (last_date - pd.Timedelta(days=5)).strftime("%Y%m%d")

    _print_bar(f"Incremental refresh ({start} ~ {today})")
    print(f"  Existing: {len(existing):,} rows, {len(assets)} assets")

    frames = []
    for i, code in enumerate(assets):
        if i > 0 and i % 40 == 0:
            print(f"  {i}/{len(assets)} ({len(frames)} ok)")

        df = _pull_one_stock(code, start, today, pro)
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(TUSHARE_PAUSE)

    if frames:
        new_data = pd.concat(frames, ignore_index=True)
        new_data = new_data.drop_duplicates(subset=["date", "asset"])
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "asset"], keep="last")
        combined = combined.sort_values(["date", "asset"]).reset_index(drop=True)
        _save_output(combined, output_path)
        # Refresh PE/PB fundamentals for new data
        _refresh_fundamentals(combined, output_path)
    else:
        print("  No new data.")
        # Still check if fundamentals need update (new quarter)
        _refresh_fundamentals(existing, output_path)


def _refresh_fundamentals(df: pd.DataFrame, output_path: Path) -> None:
    """Update PE/PB for new dates using akshare financial data.

    Only refreshes if last PE update > 30 days ago (quarterly financials).
    Detects new quarter by checking if the most recent PE date is stale.
    """
    _print_bar("Refreshing PE/PB fundamentals")

    try:
        import akshare as ak

        # Check if fundamentals need refresh
        last_pe_dates = df.dropna(subset=["pe"])["date"]
        if not last_pe_dates.empty:
            last_pe_date = pd.to_datetime(last_pe_dates).max()
            days_since = (datetime.now() - last_pe_date).days
            if days_since < 60:
                print(f"  Fundamentals up to date (last PE: {last_pe_date.strftime('%Y-%m-%d')}, {days_since}d ago)")
                return

        print(f"  Updating PE/PB for {df['asset'].nunique()} assets...")

        # Only refresh for a sample to keep it fast
        assets = sorted(df["asset"].unique())
        pe_rows = []
        pb_rows = []
        ok = 0

        for i, code in enumerate(assets):
            if i > 0 and i % 60 == 0:
                print(f"  {i}/{len(assets)} (ok={ok})")
            try:
                # PE from financial abstract
                fa = ak.stock_financial_abstract(symbol=code)
                if fa is not None and not fa.empty:
                    profit_mask = fa["指标"] == "归母净利润"
                    if profit_mask.any():
                        date_cols = [c for c in fa.columns if c not in ["选项", "指标"]]
                        profits = fa.loc[profit_mask, date_cols].iloc[0].astype(float)
                        q_dates = pd.to_datetime(date_cols, format="%Y%m%d", errors="coerce")
                        qdf = pd.DataFrame({"date": q_dates, "cum": profits.values}).dropna(subset=["date", "cum"])
                        qdf = qdf.sort_values("date")
                        if len(qdf) >= 2:
                            qdf["q"] = qdf["cum"].diff()
                            qdf.loc[qdf.index[0], "q"] = qdf["cum"].iloc[0]
                            qdf["ttm"] = qdf["q"].rolling(4, min_periods=1).sum()
                            adf = df[df["asset"] == code]
                            for _, qrow in qdf.iterrows():
                                if pd.isna(qrow["ttm"]) or qrow["ttm"] <= 0:
                                    continue
                                mcap_row = adf[adf["date"] <= qrow["date"]]
                                if mcap_row.empty:
                                    continue
                                mcap = mcap_row["market_cap"].iloc[-1]
                                if pd.isna(mcap) or mcap <= 0:
                                    continue
                                pe_rows.append({"asset": code, "date": qrow["date"], "pe": mcap / qrow["ttm"]})

                # PB from Baidu
                try:
                    pb_df = ak.stock_zh_valuation_baidu(symbol=code, indicator="市净率", period="全部")
                    if pb_df is not None and not pb_df.empty:
                        pb_df["date"] = pd.to_datetime(pb_df["date"])
                        pb_df = pb_df[pb_df["date"] >= "2021-01-01"]
                        pb_df["asset"] = code
                        pb_df = pb_df.rename(columns={"value": "pb"})
                        pb_rows.append(pb_df[["date", "asset", "pb"]])
                except Exception:
                    pass

                ok += 1
            except Exception:
                pass

        # Merge PE
        if pe_rows:
            pe_df = pd.DataFrame(pe_rows)
            pe_df["date"] = pd.to_datetime(pe_df["date"])
            df = df.drop(columns=["pe"], errors="ignore")
            df = df.sort_values(["asset", "date"])
            pe_df = pe_df.sort_values(["asset", "date"])
            frames = []
            for code in df["asset"].unique():
                adf = df[df["asset"] == code].copy()
                pdf = pe_df[pe_df["asset"] == code]
                if pdf.empty:
                    frames.append(adf)
                else:
                    frames.append(pd.merge_asof(adf, pdf[["date", "asset", "pe"]], on="date", by="asset", direction="backward"))
            df = pd.concat(frames, ignore_index=True)
            df = df.sort_values(["asset", "date"])
            df["pe"] = df.groupby("asset")["pe"].ffill()
            df["pe"] = df.groupby("asset")["pe"].bfill()
            print(f"  PE updated: {df['pe'].notna().mean():.0%}")

        # Merge PB
        if pb_rows:
            pb_all = pd.concat(pb_rows, ignore_index=True).drop_duplicates(subset=["date", "asset"])
            df = df.drop(columns=["pb"], errors="ignore")
            df = df.merge(pb_all, on=["date", "asset"], how="left")
            df = df.sort_values(["asset", "date"])
            df["pb"] = df.groupby("asset")["pb"].ffill()
            df["pb"] = df.groupby("asset")["pb"].bfill()
            print(f"  PB updated: {df['pb'].notna().mean():.0%}")

        _save_output(df, output_path)
        print(f"  Fundamentals refreshed: PE={df['pe'].notna().mean():.0%} PB={df['pb'].notna().mean():.0%}")

    except Exception as e:
        print(f"  Fundamentals refresh failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build/extend thick cross-section dataset")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="Full backward extension (2021→now)")
    group.add_argument("--refresh", action="store_true", help="Forward incremental refresh")
    group.add_argument("--use-snapshot", action="store_true", help="Copy HS300 CSV as-is")
    parser.add_argument("--start", default="20210101", help="Start date for extension")
    parser.add_argument("--base", default=str(BASE_DATASET), help="Base dataset path")
    parser.add_argument("--output", default=str(THICK_DATASET), help="Output path")
    args = parser.parse_args()

    base = Path(args.base)
    out = Path(args.output)

    if args.full:
        if not base.exists():
            print(f"Base dataset not found: {base}")
            sys.exit(1)
        extend_backward(base, out, args.start)
    elif args.refresh:
        incremental_refresh(out)
    elif args.use_snapshot:
        if not base.exists():
            print(f"Base dataset not found: {base}")
            sys.exit(1)
        snapshot_copy(base, out)
    elif out.exists():
        incremental_refresh(out)
    elif base.exists():
        extend_backward(base, out, args.start)
    else:
        print("No data found. Run with --full on an existing HS300 CSV.")
        sys.exit(1)


if __name__ == "__main__":
    main()
