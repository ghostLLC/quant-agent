"""批量拉取 HS300 成分股历史日线数据（Tushare Pro）。

用途：
  把现有 120 只 HS300 成分股的日线数据从指定起始日拉到最新，
  输出为横截面格式的 CSV，供因子发掘系统使用。

用法：
  python pull_cross_section_history.py --start 20230101
  python pull_cross_section_history.py --start 20200101 --end 20260428
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import tushare as ts


def asset_code_to_tushare(code: int | str) -> str:
    """将纯数字代码转为 Tushare 格式：000001.SZ / 600000.SH"""
    code_str = str(code).zfill(6)
    if code_str.startswith(("6",)):
        return f"{code_str}.SH"
    else:
        return f"{code_str}.SZ"


def pull_single_stock(pro, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """拉取单只股票日线，失败返回 None。"""
    try:
        df = pro.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields="ts_code,trade_date,open,high,low,close,vol,amount,pct_chg",
        )
        return df
    except Exception as e:
        print(f"  [FAIL] {ts_code}: {e}")
        return None


def pull_adj_factor(pro, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """拉取复权因子。"""
    try:
        df = pro.adj_factor(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )
        return df
    except Exception as e:
        print(f"  [ADJ FAIL] {ts_code}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="批量拉取 HS300 成分股历史日线")
    parser.add_argument("--start", default="20230101", help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default="", help="结束日期，默认今天")
    parser.add_argument("--output", default="data/hs300_cross_section_full.csv", help="输出路径")
    parser.add_argument("--metadata", default="data/hs300_cross_section_asset_metadata.csv", help="元数据路径")
    parser.add_argument("--batch-pause", type=float, default=1.5, help="每只股票间隔秒数（限速）")
    args = parser.parse_args()

    end_date = args.end or time.strftime("%Y%m%d")

    # 读取元数据
    meta_path = Path(args.metadata)
    if not meta_path.exists():
        print(f"元数据文件不存在: {meta_path}")
        sys.exit(1)

    meta = pd.read_csv(meta_path)
    print(f"元数据: {len(meta)} 只资产")

    # 初始化 Tushare
    import os
    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        print("TUSHARE_TOKEN 环境变量未设置")
        sys.exit(1)
    pro = ts.pro_api(token)

    # 如果已有输出文件，加载已成功的资产，跳过
    output_path = Path(args.output)
    already_done = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        already_done = set(existing["ts_code"].unique())
        print(f"已有数据: {len(already_done)} 只资产，跳过")

    # 逐只拉取
    all_dfs = []
    failed = []

    # 加载已有数据
    if already_done:
        all_dfs.append(pd.read_csv(output_path))

    for i, row in meta.iterrows():
        ts_code = asset_code_to_tushare(row["asset"])
        asset_name = row.get("asset_name", "")
        industry = row.get("industry", "")

        # 跳过已拉取的
        if ts_code in already_done:
            print(f"[{i + 1}/{len(meta)}] {ts_code} {asset_name}... SKIP")
            continue

        print(f"[{i + 1}/{len(meta)}] {ts_code} {asset_name}...", end=" ", flush=True)

        df = pull_single_stock(pro, ts_code, args.start, end_date)
        if df is None or len(df) == 0:
            print("NO DATA")
            failed.append(ts_code)
            continue

        # 拉复权因子
        adj_df = pull_adj_factor(pro, ts_code, args.start, end_date)
        if adj_df is not None and len(adj_df) > 0:
            df = df.merge(adj_df[["ts_code", "trade_date", "adj_factor"]], on=["ts_code", "trade_date"], how="left")
        else:
            df["adj_factor"] = 1.0

        # 添加元数据列
        df["asset"] = row["asset"]
        df["asset_name"] = asset_name
        df["industry"] = industry
        df["market_cap"] = row.get("market_cap", 0)
        df["float_market_cap"] = row.get("float_market_cap", 0)

        # 重命名列
        df = df.rename(columns={
            "vol": "volume",
            "trade_date": "date",
        })

        all_dfs.append(df)
        print(f"OK ({len(df)} rows)")

        time.sleep(args.batch_pause)

    if not all_dfs:
        print("没有任何数据拉取成功")
        sys.exit(1)

    # 合并
    result = pd.concat(all_dfs, ignore_index=True)

    # 日期格式统一
    result["date"] = pd.to_datetime(result["date"], format="mixed")

    # 排序
    result = result.sort_values(["date", "asset"]).reset_index(drop=True)

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 统计
    trading_days = result["date"].nunique()
    assets = result["asset"].nunique()
    date_min = result["date"].min()
    date_max = result["date"].max()
    span_days = (date_max - date_min).days
    span_years = span_days / 365.25

    print(f"\n=== 拉取完成 ===")
    print(f"总行数: {len(result):,}")
    print(f"交易日: {trading_days}")
    print(f"资产数: {assets}")
    print(f"日期范围: {date_min.strftime('%Y-%m-%d')} ~ {date_max.strftime('%Y-%m-%d')}")
    print(f"时间跨度: {span_days} 天 ({span_years:.1f} 年)")
    print(f"失败: {len(failed)} 只: {failed}")
    print(f"输出: {output_path}")
    print(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # 6 个月样本外评估
    from datetime import timedelta
    cutoff = date_max - timedelta(days=180)
    in_sample = result[result["date"] <= cutoff]
    out_sample = result[result["date"] > cutoff]
    print(f"\n=== 6 个月样本外拆分 ===")
    print(f"训练集: {date_min.strftime('%Y-%m-%d')} ~ {cutoff.strftime('%Y-%m-%d')} ({in_sample['date'].nunique()} 交易日)")
    print(f"测试集: {cutoff.strftime('%Y-%m-%d')} ~ {date_max.strftime('%Y-%m-%d')} ({out_sample['date'].nunique()} 交易日)")


if __name__ == "__main__":
    main()
