from pathlib import Path

import pandas as pd
import pytest

from russian_rates.analysis import (
    align_ruonia_with_key_rate,
    load_rates,
    period_statistics,
    quality_report,
)


def _data():
    ruonia = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-05", "2026-01-07", "2026-01-09"]),
        "ruonia_pct": [15.8, 15.6, 15.4],
        "volume_bln_rub": [float("nan"), 90.0, 100.0],
    })
    key_rate = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-06", "2026-01-08"]),
        "key_rate_pct": [16.0, 15.5],
    })
    zcyc = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-09"] * 12),
        "tenor_years": [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0],
        "yield_pct": [15.0] * 12,
    })
    return {"ruonia": ruonia, "key_rate": key_rate, "zcyc": zcyc}


def test_load_and_schema_validation(tmp_path):
    data = _data()
    for series, frame in data.items():
        frame.to_parquet(tmp_path / f"{series}.parquet", index=False)
    loaded = load_rates(tmp_path)
    assert loaded["ruonia"]["date"].dtype == "datetime64[ns]"
    assert len(loaded["zcyc"]) == 12
    assert pd.isna(loaded["ruonia"].loc[0, "volume_bln_rub"])


def test_missing_file_error_explains_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m russian_rates --series ruonia"):
        load_rates(tmp_path)


def test_missing_required_column_raises(tmp_path):
    data = _data()
    data["ruonia"].drop(columns="ruonia_pct").to_parquet(tmp_path / "ruonia.parquet", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_rates(tmp_path)


def test_quality_report_does_not_treat_zcyc_dates_as_duplicates():
    report = quality_report(_data()).set_index("series")
    assert report.loc["zcyc", "rows"] == 12
    assert report.loc["zcyc", "dates"] == 1
    assert report.loc["zcyc", "duplicate_keys"] == 0
    assert report.loc["zcyc", "missing_zcyc_tenors"] == 0
    assert report.loc["ruonia", "missing_aux"] == 1
    assert report.loc["ruonia", "missing_rates"] == 0


def test_quality_report_catches_missing_curve_tenors_and_duplicates():
    data = _data()
    data["zcyc"] = pd.concat(
        [data["zcyc"].iloc[:-1], data["zcyc"].iloc[[0]]],
        ignore_index=True,
    )
    report = quality_report(data).set_index("series")
    assert report.loc["zcyc", "duplicate_keys"] == 1
    assert report.loc["zcyc", "missing_zcyc_tenors"] == 1


def test_asof_join_keeps_all_ruonia_dates_and_never_uses_future_key_rate():
    d = _data()
    aligned = align_ruonia_with_key_rate(d["ruonia"], d["key_rate"])
    assert len(aligned) == 3
    assert pd.isna(aligned.loc[0, "key_rate_pct"])
    assert aligned.loc[1, "key_rate_pct"] == 16.0
    assert aligned.loc[1, "spread_pp"] == pytest.approx(-0.4)
    assert aligned.loc[2, "key_rate_pct"] == 15.5
    assert aligned.loc[2, "spread_pp"] == pytest.approx(-0.1)


def test_asof_join_requires_unique_dates():
    d = _data()
    duplicated = pd.concat([d["ruonia"], d["ruonia"].iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        align_ruonia_with_key_rate(duplicated, d["key_rate"])


def test_asof_join_empty_key_rate_is_nullable():
    d = _data()
    aligned = align_ruonia_with_key_rate(d["ruonia"], d["key_rate"].iloc[:0])
    assert len(aligned) == len(d["ruonia"])
    assert aligned["spread_pp"].isna().all()


def test_period_2023_onwards_includes_2026():
    d = _data()
    report = period_statistics(d["ruonia"], {"2023–latest": ("2023-01-01", None)})
    assert report.loc[0, "observations"] == 3
    assert report.loc[0, "mean_pct"] == pytest.approx(15.6)


def test_periods_do_not_overlap_at_year_boundaries():
    ruonia = pd.DataFrame({
        "date": pd.to_datetime(["2022-12-31", "2023-01-01"]),
        "ruonia_pct": [12.0, 13.0],
    })
    periods = {"through_2022": ("2020-01-01", "2023-01-01"),
               "since_2023": ("2023-01-01", None)}
    report = period_statistics(ruonia, periods)
    assert report["observations"].tolist() == [1, 1]
    assert report["mean_pct"].tolist() == [12.0, 13.0]
