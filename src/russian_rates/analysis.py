"""Reusable data-quality checks and exploratory analysis for CBR interest-rate data.

No missing rates are imputed. RUONIA is the primary series; it should not be
inner-joined to the ZCYC trading calendar merely to draw historical charts.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from .cbr import COLUMNS, KEYS, TENORS_YEARS

RATE_COLUMN = {"ruonia": "ruonia_pct", "key_rate": "key_rate_pct", "zcyc": "yield_pct"}


def load_rates(data_dir: str | Path = "data/processed") -> dict[str, pd.DataFrame]:
    """Read the three typed Parquet datasets, retaining their original dates."""
    root = Path(data_dir)
    result: dict[str, pd.DataFrame] = {}
    for series, required_columns in COLUMNS.items():
        path = root / f"{series}.parquet"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run: python -m russian_rates --series {series}"
            )
        frame = pd.read_parquet(path)
        absent = set(required_columns) - set(frame.columns)
        if absent:
            raise ValueError(f"{path}: missing columns {sorted(absent)}")
        frame = frame[required_columns].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        for col in required_columns:
            if col != "date":
                frame[col] = pd.to_numeric(frame[col], errors="raise")
        result[series] = frame.sort_values(KEYS[series], kind="stable").reset_index(drop=True)
    return result


def quality_report(data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Inspect raw data quality; holidays are NOT automatically treated as missing."""
    records: list[dict] = []
    for series in RATE_COLUMN:
        frame = data[series]
        keys = KEYS[series]
        date_series = frame["date"].dropna().drop_duplicates().sort_values()
        gaps = date_series.diff().dt.days.dropna()
        aux_nulls = int(frame["volume_bln_rub"].isna().sum()) if series == "ruonia" else 0
        missing_tenors = 0
        if series == "zcyc":
            for _, group in frame.dropna(subset=["date"]).groupby("date"):
                missing_tenors += len(set(TENORS_YEARS) - set(group["tenor_years"].dropna()))
        records.append({
            "series": series,
            "rows": len(frame),
            "dates": int(frame["date"].nunique()),
            "first_date": frame["date"].min() if not frame.empty else pd.NaT,
            "last_date": frame["date"].max() if not frame.empty else pd.NaT,
            "duplicate_keys": int(frame.duplicated(subset=keys).sum()),
            "missing_rates": int(frame[RATE_COLUMN[series]].isna().sum()),
            "missing_dates": int(frame["date"].isna().sum()),
            "missing_aux": aux_nulls,
            "max_gap_days": int(gaps.max()) if not gaps.empty else 0,
            "missing_zcyc_tenors": missing_tenors,
        })
    return pd.DataFrame.from_records(records)


def align_ruonia_with_key_rate(
    ruonia: pd.DataFrame, key_rate: pd.DataFrame
) -> pd.DataFrame:
    """Match each RUONIA observation to the latest available key rate at or BEFORE its date.

    This is for historical spreads, not a substitute for modelling the underlying
    policy-rate process. Never join to a later key-rate date (look-ahead).
    """
    for series, frame in (("ruonia", ruonia), ("key_rate", key_rate)):
        if frame["date"].isna().any() or frame["date"].duplicated().any():
            raise ValueError(f"{series}: null/duplicate dates prevent as-of alignment")
    left = ruonia[["date", "ruonia_pct"]].sort_values("date").reset_index(drop=True)
    right = key_rate[["date", "key_rate_pct"]].sort_values("date").reset_index(drop=True)
    if right.empty:
        aligned = left.assign(key_rate_pct=float("nan"))
    else:
        aligned = pd.merge_asof(left, right, on="date", direction="backward")
    aligned["spread_pp"] = aligned["ruonia_pct"] - aligned["key_rate_pct"]
    return aligned


def period_statistics(
    ruonia: pd.DataFrame,
    periods: Mapping[str, tuple[str, str | None]],
) -> pd.DataFrame:
    """Descriptive level statistics by regime; not Vasicek parameter estimates."""
    records: list[dict] = []
    for label, (start, end) in periods.items():
        dates = ruonia["date"]
        mask = dates.ge(pd.Timestamp(start))
        if end is not None:
            mask &= dates.lt(pd.Timestamp(end))
        sample = ruonia.loc[mask, "ruonia_pct"].dropna()
        records.append({
            "period": label,
            "observations": len(sample),
            "mean_pct": float(sample.mean()) if not sample.empty else float("nan"),
            "std_pct": float(sample.std()) if len(sample) > 1 else float("nan"),
            "min_pct": float(sample.min()) if not sample.empty else float("nan"),
            "max_pct": float(sample.max()) if not sample.empty else float("nan"),
        })
    return pd.DataFrame.from_records(records)
