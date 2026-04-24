from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import akshare as ak
import pandas as pd


ETF_SYMBOL = "510300"
ETF_SINA_SYMBOL = "sh510300"
HS300_INDEX_SYMBOL = "000300"


def _normalize_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["date", "open", "high", "low", "close", "volume"]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"ETF 数据缺少必要字段: {missing}")

    normalized = data.copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.strftime("%Y-%m-%d")
    normalized = normalized.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return normalized


def _format_date_argument(value: str | None, fallback: pd.Timestamp) -> str:
    if value:
        return pd.to_datetime(value).strftime("%Y%m%d")
    return fallback.strftime("%Y%m%d")


def _normalize_cross_section_history(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    rename_map = {
        "日期": "date",
        "股票代码": "asset",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_chg",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    normalized = data.rename(columns=rename_map).copy()
    if "asset" not in normalized.columns:
        normalized["asset"] = str(symbol)
    normalized["asset"] = normalized["asset"].astype(str).str.zfill(6)
    normalized["date"] = pd.to_datetime(normalized["date"])

    numeric_columns = [
        column
        for column in ["open", "high", "low", "close", "volume", "amount", "amplitude", "pct_chg", "change", "turnover"]
        if column in normalized.columns
    ]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    normalized = normalized.dropna(subset=["close", "volume"]).reset_index(drop=True)
    return normalized


def _normalize_refresh_report(payload: dict[str, Any]) -> dict[str, Any]:
    report = dict(payload)
    report["attempted_assets"] = int(report.get("attempted_assets", 0) or 0)
    report["succeeded_assets"] = int(report.get("succeeded_assets", 0) or 0)
    report["failed_assets"] = int(report.get("failed_assets", 0) or 0)
    report["reused_assets"] = int(report.get("reused_assets", 0) or 0)
    report["resume_used"] = bool(report.get("resume_used", False))
    report["failure_sample"] = list(report.get("failure_sample", []))[:10]
    report["failed_asset_list"] = [str(item).zfill(6) for item in report.get("failed_asset_list", [])]
    report["completed_asset_list"] = [str(item).zfill(6) for item in report.get("completed_asset_list", [])]
    report["prioritized_failed_assets"] = int(report.get("prioritized_failed_assets", 0) or 0)
    report["retry_failed_assets"] = int(report.get("retry_failed_assets", 0) or 0)
    report["industry_secondary_filled_assets"] = int(report.get("industry_secondary_filled_assets", 0) or 0)
    return report


def _read_existing_cross_section(output_path: str | Path) -> pd.DataFrame:
    output = Path(output_path)
    if not output.exists():
        return pd.DataFrame()
    df = pd.read_csv(output)
    if df.empty:
        return pd.DataFrame()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "asset" in df.columns:
        df["asset"] = df["asset"].astype(str).str.zfill(6)
    return df


def _safe_read_refresh_report(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_refresh_report(report_path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_refresh_report(payload)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def _build_refresh_report_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    return output.with_name(f"{output.stem}_refresh_report.json")


def _normalize_constituents(data: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "品种代码": "asset",
        "品种名称": "asset_name",
        "纳入日期": "in_date",
    }
    normalized = data.rename(columns=rename_map).copy()
    required_cols = ["asset", "asset_name"]
    missing = [col for col in required_cols if col not in normalized.columns]
    if missing:
        raise ValueError(f"成分股数据缺少必要字段: {missing}")
    normalized["asset"] = normalized["asset"].astype(str).str.zfill(6)
    if "in_date" in normalized.columns:
        normalized["in_date"] = pd.to_datetime(normalized["in_date"], errors="coerce")
    return normalized.sort_values("asset").drop_duplicates(subset=["asset"]).reset_index(drop=True)


def fetch_from_sina(symbol: str = ETF_SINA_SYMBOL) -> pd.DataFrame:
    df = ak.fund_etf_hist_sina(symbol=symbol)
    if df.empty:
        raise ValueError(f"新浪源未获取到 ETF 数据: {symbol}")
    return _normalize_dataframe(df)


def _build_metadata_cache_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    return output.with_name(f"{output.stem}_asset_metadata.csv")


def _read_asset_metadata_cache(cache_path: str | Path) -> pd.DataFrame:
    path = Path(cache_path)
    columns = ["asset", "asset_name", "industry", "market_cap", "float_market_cap", "total_share", "float_share", "listed_date"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=columns)
    if "asset" in df.columns:
        df["asset"] = df["asset"].astype(str).str.zfill(6)
    return df


def _clean_text_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "nat"}:
        return ""
    return text


def _is_unknown_industry(value: Any) -> bool:
    industry = _clean_text_value(value)
    return not industry or industry.lower() == "unknown"


def _has_meaningful_metadata(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    if _clean_text_value(record.get("asset_name")):
        return True
    if not _is_unknown_industry(record.get("industry")):
        return True
    if _clean_text_value(record.get("listed_date")):
        return True
    for key in ["market_cap", "float_market_cap", "total_share", "float_share"]:
        if pd.notna(pd.to_numeric(record.get(key), errors="coerce")):
            return True
    return False


def _normalize_cached_metadata_record(asset: str, record: dict[str, Any] | None) -> dict[str, Any]:
    fallback = dict(record or {})
    return {
        "asset": str(asset).zfill(6),
        "asset_name": _clean_text_value(fallback.get("asset_name")),
        "industry": _clean_text_value(fallback.get("industry")) or "unknown",
        "market_cap": pd.to_numeric(fallback.get("market_cap"), errors="coerce"),
        "float_market_cap": pd.to_numeric(fallback.get("float_market_cap"), errors="coerce"),
        "total_share": pd.to_numeric(fallback.get("total_share"), errors="coerce"),
        "float_share": pd.to_numeric(fallback.get("float_share"), errors="coerce"),
        "listed_date": _clean_text_value(fallback.get("listed_date")) or None,
    }


def _normalize_info_em_payload(symbol: str, data: pd.DataFrame) -> dict[str, Any]:

    if data.empty:
        raise ValueError(f"未获取到个股资料: {symbol}")
    value_map = {
        str(row["item"]): row["value"]
        for _, row in data.iterrows()
        if "item" in data.columns and "value" in data.columns
    }
    industry = str(value_map.get("行业") or "unknown").strip() or "unknown"
    market_cap = pd.to_numeric(value_map.get("总市值"), errors="coerce")
    float_market_cap = pd.to_numeric(value_map.get("流通市值"), errors="coerce")
    total_share = pd.to_numeric(value_map.get("总股本"), errors="coerce")
    float_share = pd.to_numeric(value_map.get("流通股"), errors="coerce")
    listed_date = str(value_map.get("上市时间") or "").strip()
    if listed_date:
        listed_date = pd.to_datetime(listed_date, format="%Y%m%d", errors="coerce")
        listed_date = None if pd.isna(listed_date) else listed_date.strftime("%Y-%m-%d")
    return {
        "asset": str(symbol).zfill(6),
        "asset_name": str(value_map.get("股票简称") or "").strip(),
        "industry": industry,
        "market_cap": None if pd.isna(market_cap) else float(market_cap),
        "float_market_cap": None if pd.isna(float_market_cap) else float(float_market_cap),
        "total_share": None if pd.isna(total_share) else float(total_share),
        "float_share": None if pd.isna(float_share) else float(float_share),
        "listed_date": listed_date,
    }


def _normalize_xq_payload(symbol: str, data: pd.DataFrame) -> dict[str, Any]:
    if data.empty:
        raise ValueError(f"未获取到雪球个股资料: {symbol}")
    value_map = {
        str(row["item"]): row["value"]
        for _, row in data.iterrows()
        if "item" in data.columns and "value" in data.columns
    }
    affiliate_industry = value_map.get("affiliate_industry") or {}
    industry = "unknown"
    if isinstance(affiliate_industry, dict):
        industry = _clean_text_value(affiliate_industry.get("ind_name")) or "unknown"
    listed_date = _clean_text_value(value_map.get("list_date")) or _clean_text_value(value_map.get("listed_date"))
    if listed_date:
        listed_date = pd.to_datetime(listed_date, errors="coerce")
        listed_date = None if pd.isna(listed_date) else listed_date.strftime("%Y-%m-%d")
    return {
        "asset": str(symbol).zfill(6),
        "asset_name": _clean_text_value(value_map.get("org_short_name_cn")) or _clean_text_value(value_map.get("stock_name")),
        "industry": industry,
        "market_cap": None,
        "float_market_cap": None,
        "total_share": None,
        "float_share": None,
        "listed_date": listed_date,
    }


def _fetch_stock_metadata_from_xq(symbol: str, max_retries: int = 2, retry_delay: float = 1.0) -> dict[str, Any]:
    market_prefix = "SH" if str(symbol).zfill(6).startswith("6") else "SZ"
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_individual_basic_info_xq(symbol=f"{market_prefix}{str(symbol).zfill(6)}")
            return _normalize_xq_payload(symbol, df)
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(retry_delay)
    raise RuntimeError(f"雪球个股资料抓取失败，已重试 {max_retries} 次: {symbol}") from last_error


def fetch_stock_metadata(symbol: str, max_retries: int = 3, retry_delay: float = 1.0) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_individual_info_em(symbol=str(symbol).zfill(6))
            return _normalize_info_em_payload(symbol, df)
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(retry_delay)
    try:
        xq_record = _fetch_stock_metadata_from_xq(symbol, max_retries=2, retry_delay=retry_delay)
        xq_record["metadata_source"] = "xq_basic_info"
        return xq_record
    except Exception as fallback_exc:
        raise RuntimeError(f"个股资料抓取失败，已重试 {max_retries} 次: {symbol}") from (last_error or fallback_exc)



def _merge_metadata_record(base: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    merged = _normalize_cached_metadata_record(base.get("asset") or fallback.get("asset") or "", base)
    normalized_fallback = _normalize_cached_metadata_record(merged.get("asset") or "", fallback)
    for key, value in normalized_fallback.items():
        if key == "asset":
            continue
        current = merged.get(key)
        if key in {"market_cap", "float_market_cap", "total_share", "float_share"}:
            current_numeric = pd.to_numeric(current, errors="coerce")
            incoming_numeric = pd.to_numeric(value, errors="coerce")
            if pd.isna(current_numeric) and pd.notna(incoming_numeric):
                merged[key] = float(incoming_numeric)
            continue
        if key == "industry":
            if _is_unknown_industry(current) and not _is_unknown_industry(value):
                merged[key] = _clean_text_value(value)
            continue
        if key == "listed_date":
            if not _clean_text_value(current) and _clean_text_value(value):
                merged[key] = _clean_text_value(value)
            continue
        if not _clean_text_value(current) and _clean_text_value(value):
            merged[key] = _clean_text_value(value)
    merged["industry"] = _clean_text_value(merged.get("industry")) or "unknown"
    merged["asset_name"] = _clean_text_value(merged.get("asset_name"))
    merged["listed_date"] = _clean_text_value(merged.get("listed_date")) or None
    return merged



def _fetch_sina_industry_map(
    target_assets: list[str] | None = None,
    max_retries: int = 2,
    retry_delay: float = 1.5,
) -> tuple[dict[str, str], dict[str, Any]]:
    normalized_targets = {str(item).zfill(6) for item in (target_assets or [])}
    industry_map: dict[str, str] = {}
    scanned_sectors = 0
    sector_errors: list[dict[str, str]] = []
    spot_df = ak.stock_sector_spot(indicator="新浪行业")
    if spot_df.empty:
        raise ValueError("新浪行业板块列表为空")

    for row in spot_df.itertuples(index=False):
        sector_label = str(getattr(row, "label", "") or "").strip()
        sector_name = str(getattr(row, "板块", "") or "").strip()
        if not sector_label or not sector_name:
            continue
        scanned_sectors += 1
        last_error: Exception | None = None
        detail_df = pd.DataFrame()
        for attempt in range(1, max_retries + 1):
            try:
                detail_df = ak.stock_sector_detail(sector=sector_label)
                break
            except Exception as exc:
                last_error = exc
                if attempt == max_retries:
                    sector_errors.append({"sector": sector_name, "label": sector_label, "error": str(exc)})
                else:
                    time.sleep(retry_delay)
        if detail_df.empty:
            continue
        if "code" not in detail_df.columns:
            continue
        codes = detail_df["code"].astype(str).str.zfill(6)
        for code in codes:
            if normalized_targets and code not in normalized_targets:
                continue
            industry_map.setdefault(code, sector_name)
        if normalized_targets and normalized_targets.issubset(industry_map.keys()):
            break

    report = {
        "industry_secondary_source": "sina_sector",
        "industry_secondary_scanned_sectors": int(scanned_sectors),
        "industry_secondary_matched_assets": int(len(industry_map)),
        "industry_secondary_failure_sample": sector_errors[:10],
    }
    return industry_map, report


def build_asset_metadata_map(
    assets: list[str],
    asset_name_map: dict[str, str] | None = None,
    existing_cache: pd.DataFrame | None = None,
    pause_seconds: float = 0.1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized_assets = [str(item).zfill(6) for item in assets]
    normalized_asset_name_map = {
        str(asset).zfill(6): _clean_text_value(name)
        for asset, name in (asset_name_map or {}).items()
        if _clean_text_value(name)
    }
    cache = existing_cache.copy() if existing_cache is not None and not existing_cache.empty else pd.DataFrame()
    if not cache.empty and "asset" in cache.columns:
        cache["asset"] = cache["asset"].astype(str).str.zfill(6)
        cache = cache.drop_duplicates(subset=["asset"], keep="last")
    cache_map = cache.set_index("asset").to_dict(orient="index") if not cache.empty and "asset" in cache.columns else {}

    records: list[dict[str, Any]] = []
    metadata_errors: list[dict[str, str]] = []
    reused_assets = 0
    asset_name_secondary_filled_assets = 0
    xq_fallback_assets = 0
    for idx, asset in enumerate(normalized_assets, start=1):
        cached = _normalize_cached_metadata_record(asset, cache_map.get(asset)) if asset in cache_map else None
        should_refresh = (not _has_meaningful_metadata(cached)) or _is_unknown_industry((cached or {}).get("industry"))
        if not should_refresh and cached is not None:
            records.append(dict(cached))
            reused_assets += 1
        else:
            try:
                fresh = fetch_stock_metadata(asset)
                if _clean_text_value(fresh.get("metadata_source")) == "xq_basic_info":
                    xq_fallback_assets += 1
                record = _merge_metadata_record(fresh, dict(cached) if cached else {})
                record["asset"] = asset
                records.append(record)
            except Exception as exc:
                metadata_errors.append({"asset": asset, "error": str(exc)})
                records.append(_normalize_cached_metadata_record(asset, cached))
        if idx < len(normalized_assets) and pause_seconds > 0:
            time.sleep(pause_seconds)



    metadata_df = pd.DataFrame(records).drop_duplicates(subset=["asset"], keep="last") if records else pd.DataFrame(columns=["asset"])
    secondary_report: dict[str, Any] = {
        "industry_secondary_source": None,
        "industry_secondary_scanned_sectors": 0,
        "industry_secondary_matched_assets": 0,
        "industry_secondary_failure_sample": [],
    }
    secondary_filled_assets = 0
    if not metadata_df.empty:
        metadata_df["asset"] = metadata_df["asset"].astype(str).str.zfill(6)
        metadata_df["asset_name"] = metadata_df["asset_name"].map(_clean_text_value)
        metadata_df["industry"] = metadata_df["industry"].map(lambda value: _clean_text_value(value) or "unknown")
        metadata_df["listed_date"] = metadata_df["listed_date"].map(lambda value: _clean_text_value(value) or None)
        for numeric_column in ["market_cap", "float_market_cap", "total_share", "float_share"]:
            if numeric_column in metadata_df.columns:
                metadata_df[numeric_column] = pd.to_numeric(metadata_df[numeric_column], errors="coerce")
        missing_name_mask = metadata_df["asset_name"].eq("")
        if normalized_asset_name_map and bool(missing_name_mask.any()):
            fillable_name_mask = missing_name_mask & metadata_df["asset"].isin(normalized_asset_name_map.keys())
            asset_name_secondary_filled_assets = int(fillable_name_mask.sum())
            metadata_df.loc[fillable_name_mask, "asset_name"] = metadata_df.loc[fillable_name_mask, "asset"].map(normalized_asset_name_map)
        missing_industry_assets = metadata_df.loc[
            metadata_df["industry"].map(_is_unknown_industry),
            "asset",
        ].astype(str).str.zfill(6).tolist()
        if missing_industry_assets:
            try:
                industry_map, secondary_report = _fetch_sina_industry_map(missing_industry_assets)
                if industry_map:
                    mask = metadata_df["asset"].isin(industry_map.keys()) & metadata_df["industry"].map(_is_unknown_industry)
                    secondary_filled_assets = int(mask.sum())
                    metadata_df.loc[mask, "industry"] = metadata_df.loc[mask, "asset"].map(industry_map)
            except Exception as exc:
                secondary_report = {
                    "industry_secondary_source": "sina_sector",
                    "industry_secondary_scanned_sectors": 0,
                    "industry_secondary_matched_assets": 0,
                    "industry_secondary_failure_sample": [{"error": str(exc)}],
                }

    report = {
        "requested_assets": int(len(normalized_assets)),
        "metadata_reused_assets": int(reused_assets),
        "metadata_failed_assets": int(len(metadata_errors)),
        "metadata_failure_sample": metadata_errors[:10],
        "metadata_xq_fallback_assets": int(xq_fallback_assets),
        "asset_name_secondary_source": "constituents" if normalized_asset_name_map else None,
        "asset_name_secondary_filled_assets": int(asset_name_secondary_filled_assets),
        "industry_secondary_filled_assets": int(secondary_filled_assets),
        **secondary_report,
    }

    return metadata_df, report




def _to_akshare_daily_symbol(symbol: str) -> str:
    normalized = str(symbol).strip().lower()
    if normalized.startswith(("sh", "sz", "bj")):
        return normalized
    code = normalized.zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("4", "8")):
        return f"bj{code}"
    return f"sz{code}"


def _fetch_stock_history_frame_via_daily(
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    adjust: str,
) -> pd.DataFrame:
    daily_symbol = _to_akshare_daily_symbol(symbol)
    df = ak.stock_zh_a_daily(symbol=daily_symbol, adjust=adjust)
    if df.empty:
        raise ValueError(f"未获取到 A 股日线数据: {symbol}")
    normalized = df.copy()
    if "date" not in normalized.columns:
        normalized = normalized.reset_index().rename(columns={normalized.index.name or "index": "date"})
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized = normalized[(normalized["date"] >= start_ts) & (normalized["date"] <= end_ts)].reset_index(drop=True)
    if normalized.empty:
        raise ValueError(f"A 股日线数据在目标区间内为空: {symbol}")
    normalized["asset"] = str(symbol).zfill(6)
    return normalized


def fetch_from_eastmoney(symbol: str = ETF_SYMBOL, max_retries: int = 3, retry_delay: float = 2.0) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.fund_etf_hist_em(symbol=symbol, period="daily", adjust="qfq")
            if df.empty:
                raise ValueError(f"未获取到 ETF 数据: {symbol}")
            data = df.rename(
                columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                    "振幅": "amplitude",
                    "涨跌幅": "pct_chg",
                    "涨跌额": "change",
                    "换手率": "turnover",
                }
            )
            return _normalize_dataframe(data)
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                raise RuntimeError(f"东方财富源抓取 ETF 数据失败，已重试 {max_retries} 次") from exc
            time.sleep(retry_delay)
    raise RuntimeError("东方财富源抓取 ETF 数据失败") from last_error


def fetch_hs300_etf_history() -> pd.DataFrame:
    try:
        return fetch_from_sina()
    except Exception:
        return fetch_from_eastmoney()


def update_hs300_etf_csv(output_path: str | Path) -> pd.DataFrame:
    data = fetch_hs300_etf_history()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False, encoding="utf-8-sig")
    return data


def fetch_hs300_constituents(index_symbol: str = HS300_INDEX_SYMBOL) -> pd.DataFrame:
    df = ak.index_stock_cons(symbol=index_symbol)
    if df.empty:
        raise ValueError(f"未获取到沪深300成分股列表: {index_symbol}")
    return _normalize_constituents(df)


def fetch_stock_history_frame(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    adjust: str = "qfq",
    max_retries: int = 3,
    retry_delay: float = 1.5,
) -> pd.DataFrame:
    end_ts = pd.Timestamp.today().normalize()
    start_ts = end_ts - pd.Timedelta(days=365)
    formatted_start = _format_date_argument(start_date, start_ts)
    formatted_end = _format_date_argument(end_date, end_ts)
    query_start_ts = pd.to_datetime(formatted_start)
    query_end_ts = pd.to_datetime(formatted_end)
    last_error: Exception | None = None

    try:
        daily_df = _fetch_stock_history_frame_via_daily(
            symbol=symbol,
            start_ts=query_start_ts,
            end_ts=query_end_ts,
            adjust=adjust,
        )
        return _normalize_cross_section_history(daily_df, str(symbol).zfill(6))
    except Exception as exc:
        last_error = exc

    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_zh_a_hist(
                symbol=str(symbol).zfill(6),
                period="daily",
                start_date=formatted_start,
                end_date=formatted_end,
                adjust=adjust,
            )
            if df.empty:
                raise ValueError(f"未获取到 A 股历史数据: {symbol}")
            return _normalize_cross_section_history(df, str(symbol).zfill(6))
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(retry_delay)
    raise RuntimeError(f"A 股历史数据抓取失败，已重试 {max_retries} 次且后备接口不可用: {symbol}") from last_error


def build_hs300_cross_section_dataset(
    start_date: str | None = None,
    end_date: str | None = None,
    max_assets: int | None = None,
    index_symbol: str = HS300_INDEX_SYMBOL,
    pause_seconds: float = 0.2,
    existing_data: pd.DataFrame | None = None,
    resume: bool = True,
    existing_metadata: pd.DataFrame | None = None,
    priority_assets: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    constituents = fetch_hs300_constituents(index_symbol=index_symbol)
    if max_assets is not None:
        constituents = constituents.head(max_assets)

    target_assets = constituents["asset"].astype(str).str.zfill(6).tolist()
    asset_name_map = (
        constituents.assign(asset=constituents["asset"].astype(str).str.zfill(6))
        .set_index("asset")["asset_name"]
        .astype(str)
        .to_dict()
    )
    priority_assets = [asset for asset in [str(item).zfill(6) for item in (priority_assets or [])] if asset in target_assets]

    if priority_assets:
        priority_rank = {asset: idx for idx, asset in enumerate(priority_assets)}
        constituents = constituents.assign(
            _priority=constituents["asset"].astype(str).str.zfill(6).map(priority_rank).fillna(len(priority_rank) + 1)
        ).sort_values(["_priority", "asset"]).drop(columns=["_priority"]).reset_index(drop=True)

    existing = existing_data.copy() if existing_data is not None and not existing_data.empty else pd.DataFrame()
    if not existing.empty:
        if "asset" in existing.columns:
            existing["asset"] = existing["asset"].astype(str).str.zfill(6)
        if "date" in existing.columns:
            existing["date"] = pd.to_datetime(existing["date"], errors="coerce")

    metadata_df, metadata_report = build_asset_metadata_map(
        target_assets,
        asset_name_map=asset_name_map,
        existing_cache=existing_metadata,
        pause_seconds=min(max(float(pause_seconds), 0.08), 0.3),
    )

    metadata_lookup = metadata_df.set_index("asset").to_dict(orient="index") if not metadata_df.empty and "asset" in metadata_df.columns else {}

    completed_assets = set()
    reused_assets = 0
    frames: list[pd.DataFrame] = []
    errors: list[dict[str, Any]] = []
    retried_assets = 0

    if resume and not existing.empty and {"asset", "date", "close", "volume"}.issubset(existing.columns):
        existing = existing[existing["asset"].isin(target_assets)].copy()
        if start_date:
            existing = existing[existing["date"] >= pd.to_datetime(start_date)]
        if end_date:
            existing = existing[existing["date"] <= pd.to_datetime(end_date)]
        if not existing.empty:
            counts = existing.groupby("asset")["date"].nunique()
            completed_assets = {asset for asset, count in counts.items() if int(count) > 0}
            if completed_assets:
                frames.append(existing)
                reused_assets = len(completed_assets)

    def _attach_metadata(history: pd.DataFrame, asset: str, asset_name: str) -> pd.DataFrame:
        metadata = metadata_lookup.get(asset, {})
        history = history.copy()
        history["asset_name"] = str(metadata.get("asset_name") or asset_name).strip() or asset_name
        history["index_symbol"] = index_symbol
        history["industry"] = str(metadata.get("industry") or "unknown").strip() or "unknown"
        market_cap_value = pd.to_numeric(metadata.get("market_cap"), errors="coerce")
        float_market_cap_value = pd.to_numeric(metadata.get("float_market_cap"), errors="coerce")
        total_share_value = pd.to_numeric(metadata.get("total_share"), errors="coerce")
        float_share_value = pd.to_numeric(metadata.get("float_share"), errors="coerce")
        if pd.notna(total_share_value) and total_share_value > 0:
            history["market_cap"] = pd.to_numeric(history["close"], errors="coerce") * float(total_share_value)
        elif pd.notna(market_cap_value) and len(history) > 0:
            history["market_cap"] = pd.Series(float(market_cap_value), index=history.index)
        else:
            history["market_cap"] = pd.to_numeric(history["close"], errors="coerce") * pd.to_numeric(history["volume"], errors="coerce")
        if pd.notna(float_share_value) and float_share_value > 0:
            history["float_market_cap"] = pd.to_numeric(history["close"], errors="coerce") * float(float_share_value)
        elif pd.notna(float_market_cap_value) and len(history) > 0:
            history["float_market_cap"] = pd.Series(float(float_market_cap_value), index=history.index)
        history["total_share"] = total_share_value
        history["float_share"] = float_share_value
        history["listed_date"] = metadata.get("listed_date")
        return history

    def _collect_history(source_df: pd.DataFrame) -> list[dict[str, Any]]:
        failed: list[dict[str, Any]] = []
        rows = source_df.reset_index(drop=True)
        total_rows = len(rows)
        for idx, row in enumerate(rows.itertuples(index=False), start=1):
            asset_key = str(row.asset).zfill(6)
            try:
                history = fetch_stock_history_frame(asset_key, start_date=start_date, end_date=end_date)
                frames.append(_attach_metadata(history, asset_key, getattr(row, "asset_name", asset_key)))
                completed_assets.add(asset_key)
            except Exception as exc:
                failed.append({"asset": asset_key, "error": str(exc)})
            if idx < total_rows and pause_seconds > 0:
                time.sleep(pause_seconds)
        return failed

    pending = constituents[~constituents["asset"].isin(completed_assets)].reset_index(drop=True)
    errors = _collect_history(pending)
    if errors:
        retry_assets = [item["asset"] for item in errors]
        retry_df = constituents[constituents["asset"].isin(retry_assets)].copy()
        errors = [item for item in errors if item["asset"] not in set(retry_assets)]
        retried_assets = len(retry_df)
        if not retry_df.empty and pause_seconds > 0:
            time.sleep(max(0.5, pause_seconds))
        retry_errors = _collect_history(retry_df)
        errors.extend(retry_errors)

    if not frames:
        details = errors[:5]
        raise RuntimeError(f"未能成功抓取任何沪深300成分股历史数据，示例错误: {details}")

    dataset = pd.concat(frames, ignore_index=True)
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset["asset"] = dataset["asset"].astype(str).str.zfill(6)
    dataset = dataset.sort_values(["date", "asset"]).drop_duplicates(subset=["date", "asset"]).reset_index(drop=True)
    report = {
        "index_symbol": index_symbol,
        "attempted_assets": int(len(target_assets)),
        "succeeded_assets": int(len(completed_assets)),
        "failed_assets": int(len(errors)),
        "reused_assets": int(reused_assets),
        "resume_used": bool(resume and reused_assets > 0),
        "prioritized_failed_assets": int(len(priority_assets)),
        "retry_failed_assets": int(retried_assets),
        "completed_asset_list": sorted(completed_assets),
        "failed_asset_list": [item["asset"] for item in errors],
        "failure_sample": errors[:10],
        **metadata_report,
    }
    return dataset, _normalize_refresh_report(report), metadata_df


def update_hs300_cross_section_csv(
    output_path: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    max_assets: int | None = None,
    index_symbol: str = HS300_INDEX_SYMBOL,
    pause_seconds: float = 0.2,
    resume: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    existing = _read_existing_cross_section(output_path) if resume else pd.DataFrame()
    report_path = _build_refresh_report_path(output_path)
    metadata_cache_path = _build_metadata_cache_path(output_path)
    existing_metadata = _read_asset_metadata_cache(metadata_cache_path)
    previous_report = _safe_read_refresh_report(report_path)
    priority_assets = previous_report.get("failed_asset_list", []) if previous_report else []
    data, report, metadata_df = build_hs300_cross_section_dataset(
        start_date=start_date,
        end_date=end_date,
        max_assets=max_assets,
        index_symbol=index_symbol,
        pause_seconds=pause_seconds,
        existing_data=existing,
        resume=resume,
        existing_metadata=existing_metadata,
        priority_assets=priority_assets,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False, encoding="utf-8-sig")
    if not metadata_df.empty:
        metadata_df.to_csv(metadata_cache_path, index=False, encoding="utf-8-sig")
    report.update(
        {
            "output_path": str(output),
            "report_path": str(report_path),
            "metadata_cache_path": str(metadata_cache_path),
            "start_date": data["date"].min().strftime("%Y-%m-%d") if not data.empty else None,
            "end_date": data["date"].max().strftime("%Y-%m-%d") if not data.empty else None,
            "rows": int(len(data)),
            "previous_failed_assets": int(len(previous_report.get("failed_asset_list", []))) if previous_report else 0,
            "industry_coverage": round(float((data["industry"].fillna("unknown").astype(str) != "unknown").mean()), 4) if not data.empty and "industry" in data.columns else 0.0,
            "market_cap_coverage": round(float(pd.to_numeric(data.get("market_cap"), errors="coerce").notna().mean()), 4) if not data.empty and "market_cap" in data.columns else 0.0,
        }
    )
    report = _write_refresh_report(report_path, report)
    return data, report
