"""Smoke-test the full Jupyter EDA workflow using synthetic Parquet, no CBR network."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from russian_rates.cbr import TENORS_YEARS


def test_notebook_run_all_on_synthetic_data(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks" / "01_rates_analysis.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(not c.get("outputs") for c in notebook["cells"] if c["cell_type"] == "code")

    dates = pd.to_datetime([
        "2014-06-03", "2014-12-16", "2020-04-01",
        "2022-03-01", "2023-05-01", "2026-11-20",
    ])
    ruonia = pd.DataFrame({
        "date": dates,
        "ruonia_pct": [8.0, 20.0, 6.0, 19.0, 8.0, 15.0],
        "volume_bln_rub": [20.0, 40.0, float("nan"), 100.0, 75.0, 90.0],
    })
    key_rate = pd.DataFrame({
        "date": pd.to_datetime([
            "2014-06-02", "2014-12-15", "2020-03-31",
            "2022-02-28", "2023-04-30", "2026-11-19",
        ]),
        "key_rate_pct": [8.0, 17.0, 6.0, 20.0, 8.0, 16.0],
    })
    zcyc = pd.DataFrame(
        [(date, tenor, 10.0 + float(tenor) / 10) for date in dates for tenor in TENORS_YEARS],
        columns=["date", "tenor_years", "yield_pct"],
    )

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='temporary-fixture'\n")
    for name, frame in (("ruonia", ruonia), ("key_rate", key_rate), ("zcyc", zcyc)):
        frame.to_parquet(data_dir / f"{name}.parquet", index=False)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    scope = {"display": lambda _: None}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(compile(source, "01_rates_analysis.ipynb", "exec"), scope)

    aligned = scope["aligned"]
    assert len(aligned) == len(ruonia)
    assert aligned["key_rate_pct"].notna().all()
    assert scope["PERIODS"]["2023 — последняя дата"][1] is None
