from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from russian_rates.cbr import (
    CbrFormatError, TENORS_YEARS, fetch_series, parse_date, parse_table, save_parquet,
)


def html_table(headers: str, body: str) -> str:
    return f'<html><div class="table-wrapper table"><table>{headers}{body}</table></div></html>'


RUONIA_HTML = html_table(
    '<tr><th>Дата ставки</th><th>Ставка RUONIA</th><th>Объем сделок</th></tr>',
    '<tr><td>05.11.2026</td><td>14,50</td><td>1\u00a0008,21</td><td>84</td></tr>'
    '<tr><td>06.11.2026</td><td>14,40</td><td>971,22</td><td>75</td></tr>',
)
KEY_RATE_HTML = html_table(
    '<tr><th>Дата</th><th>Ставка</th></tr>',
    '<tr><td>05.11.2026</td><td>15,00</td></tr>',
)
ZCYC_HTML = html_table(
    '<tr><th rowspan="2">Дата</th><th colspan="12">Срок до погашения, лет</th></tr>'
    '<tr>' + ''.join(f'<th>{str(x).replace(".", ",")}</th>' for x in TENORS_YEARS) + '</tr>',
    '<tr><td><a href="/whatever">05.11.2026</a></td>'
    + ''.join(f'<td>{str(12 + x / 10).replace(".", ",")}</td>' for x in TENORS_YEARS)
    + '</tr>',
)


def test_parse_ruonia_numeric_dates_and_volume():
    df = parse_table(RUONIA_HTML, "ruonia", date(2026, 11, 5), date(2026, 11, 6))
    assert list(df.columns) == ["date", "ruonia_pct", "volume_bln_rub"]
    assert df["date"].dtype == "datetime64[ns]"
    assert df["ruonia_pct"].dtype == "float64"
    assert df.iloc[0]["volume_bln_rub"] == 1008.21
    assert df.iloc[0]["ruonia_pct"] == 14.5


def test_parse_key_rate():
    df = parse_table(KEY_RATE_HTML, "key_rate", date(2026, 11, 5), date(2026, 11, 6))
    assert len(df) == 1
    assert df.iloc[0]["key_rate_pct"] == 15.0


def test_parse_zcyc_12_tenors_including_20_year():
    df = parse_table(ZCYC_HTML, "zcyc", date(2026, 11, 5), date(2026, 11, 6))
    assert len(df) == 12
    assert df["tenor_years"].tolist() == list(TENORS_YEARS)
    assert df.loc[df["tenor_years"].eq(20), "yield_pct"].iloc[0] == 14.0
    assert df.loc[df["tenor_years"].eq(30), "yield_pct"].iloc[0] == 15.0


def test_exclude_rows_outside_requested_dates():
    df = parse_table(RUONIA_HTML, "ruonia", date(2026, 11, 6), date(2026, 11, 6))
    assert len(df) == 1
    assert df.iloc[0]["ruonia_pct"] == 14.4


def test_invalid_row_fails_instead_of_writing_corrupt_data():
    broken = html_table('<tr><th>Дата</th></tr>', '<tr><td>05.11.2026</td><td>14,2</td></tr>')
    with pytest.raises(CbrFormatError, match="expected at least 13"):
        parse_table(broken, "zcyc", date(2026, 11, 5), date(2026, 11, 6))


def test_invalid_number_fails():
    broken = RUONIA_HTML.replace("14,50", "error")
    with pytest.raises(CbrFormatError, match="Invalid numeric value"):
        parse_table(broken, "ruonia", date(2026, 11, 5), date(2026, 11, 6))


def test_missing_table_fails():
    with pytest.raises(CbrFormatError, match="No HTML table"):
        parse_table('<div>bot blocked</div>', "ruonia", date(2026, 11, 5), date(2026, 11, 6))


def test_parse_date():
    assert parse_date("05.11.2026") == parse_date("2026-11-05")
    with pytest.raises(ValueError, match="Invalid date"):
        parse_date("2026/11/05")


class Response:
    status_code = 200

    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, response_html):
        self.calls = []
        self.response_html = response_html

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response(self.response_html)


def test_split_into_calendar_years_and_respect_exact_ranges():
    body = (
        '<tr><td>31.12.2025</td><td>15,00</td></tr>'
        '<tr><td>02.01.2026</td><td>14,50</td></tr>'
    )
    session = FakeSession(html_table('<tr><th>Дата</th></tr>', body))
    frame = fetch_series("key_rate", "2025-12-30", "2026-01-02", session=session, pause_seconds=0)
    assert len(session.calls) == 2
    assert session.calls[0][1]["params"]["UniDbQuery.From"] == "30.12.2025"
    assert session.calls[0][1]["params"]["UniDbQuery.To"] == "31.12.2025"
    assert session.calls[1][1]["params"]["UniDbQuery.From"] == "01.01.2026"
    assert session.calls[1][1]["params"]["UniDbQuery.To"] == "02.01.2026"
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-12-31", "2026-01-02"]
    assert session.calls[0][1]["timeout"] == (10, 60)


def test_empty_date_range_fails():
    session = FakeSession(KEY_RATE_HTML)
    with pytest.raises(ValueError, match="No key_rate observations"):
        fetch_series("key_rate", "2026-12-01", "2026-12-02", session=session, pause_seconds=0)


def test_out_of_order_date_range_fails_without_request():
    session = FakeSession(KEY_RATE_HTML)
    with pytest.raises(ValueError, match="Start date"):
        fetch_series("key_rate", "2026-12-02", "2026-12-01", session=session)
    assert not session.calls


def test_duplicate_date_rejected():
    doubled = RUONIA_HTML.replace(
        '</table>', '<tr><td>05.11.2026</td><td>14,99</td><td>1,0</td></tr></table>'
    )
    session = FakeSession(doubled)
    with pytest.raises(ValueError, match="Duplicate"):
        fetch_series("ruonia", "2026-11-05", "2026-11-06", session=session, pause_seconds=0)


def test_parquet_upsert_uses_latest_data_and_atomic_rename(monkeypatch, tmp_path):
    path = tmp_path / "ruonia.parquet"
    path.write_bytes(b"old fake parquet")
    prior = pd.DataFrame({
        "date": pd.to_datetime(["2026-11-04", "2026-11-05"]),
        "ruonia_pct": [14.1, 14.2],
        "volume_bln_rub": [800.0, 810.0],
    })
    updated = pd.DataFrame({
        "date": pd.to_datetime(["2026-11-05", "2026-11-06"]),
        "ruonia_pct": [14.5, 14.6],
        "volume_bln_rub": [1000.0, 900.0],
    })
    persisted = {}
    monkeypatch.setattr(pd, "read_parquet", lambda _: prior)

    def mock_to_parquet(self, destination, **kwargs):
        persisted["frame"] = self.copy()
        persisted["kwargs"] = kwargs
        Path(destination).write_bytes(b"new fake parquet")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", mock_to_parquet)
    saved_path = save_parquet(updated, "ruonia", tmp_path)
    assert saved_path == path
    assert persisted["frame"]["ruonia_pct"].tolist() == [14.1, 14.5, 14.6]
    assert persisted["kwargs"] == {"engine": "pyarrow", "index": False}
    assert path.read_bytes() == b"new fake parquet"
    assert not (tmp_path / ".ruonia.parquet.tmp").exists()


def test_parquet_replace_does_not_read_old_file(monkeypatch, tmp_path):
    (tmp_path / "key_rate.parquet").write_bytes(b"old")
    monkeypatch.setattr(pd, "read_parquet", lambda _: pytest.fail("read_parquet called"))
    monkeypatch.setattr(
        pd.DataFrame, "to_parquet",
        lambda self, path, **kw: Path(path).write_bytes(b"new")
    )
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-11-06"]), "key_rate_pct": [15.0]
    })
    save_parquet(frame, "key_rate", tmp_path, replace=True)
    assert (tmp_path / "key_rate.parquet").read_bytes() == b"new"

@pytest.mark.parametrize("missing_marker", ["—", "–", "-", ""])
def test_ruonia_missing_volume_does_not_erase_valid_rate(missing_marker):
    html = RUONIA_HTML.replace("1\u00a0008,21", missing_marker)
    parsed = parse_table(html, "ruonia", date(2026, 11, 5), date(2026, 11, 6))
    assert len(parsed) == 2
    assert parsed.loc[0, "ruonia_pct"] == 14.5
    assert pd.isna(parsed.loc[0, "volume_bln_rub"])
    assert parsed.loc[1, "volume_bln_rub"] == 971.22


def test_ruonia_unknown_volume_text_is_not_silently_accepted():
    html = RUONIA_HTML.replace("1\u00a0008,21", "unknown")
    with pytest.raises(CbrFormatError, match="Invalid numeric value"):
        parse_table(html, "ruonia", date(2026, 11, 5), date(2026, 11, 6))


def test_missing_interest_rate_is_still_an_error():
    html = RUONIA_HTML.replace("14,50", "—")
    with pytest.raises(CbrFormatError, match="Invalid numeric value"):
        parse_table(html, "ruonia", date(2026, 11, 5), date(2026, 11, 6))


def test_missing_ruonia_volume_roundtrips_in_real_parquet(tmp_path):
    html = RUONIA_HTML.replace("1\u00a0008,21", "—")
    parsed = parse_table(html, "ruonia", date(2026, 11, 5), date(2026, 11, 6))
    output = save_parquet(parsed, "ruonia", tmp_path)
    loaded = pd.read_parquet(output)
    assert pd.isna(loaded.loc[0, "volume_bln_rub"])
    assert loaded.loc[0, "ruonia_pct"] == 14.5
    assert loaded["ruonia_pct"].dtype == "float64"
    assert loaded["volume_bln_rub"].dtype == "float64"


def test_existing_parquet_with_missing_ruonia_volume_can_be_updated(tmp_path):
    initial_html = RUONIA_HTML.replace("1\u00a0008,21", "—")
    original = parse_table(initial_html, "ruonia", date(2026, 11, 5), date(2026, 11, 6))
    save_parquet(original, "ruonia", tmp_path)
    next_day_html = html_table(
        '<tr><th>Дата</th><th>Ставка</th><th>Объём</th></tr>',
        '<tr><td>07.11.2026</td><td>14,30</td><td>—</td></tr>',
    )
    additional = parse_table(next_day_html, "ruonia", date(2026, 11, 7), date(2026, 11, 7))
    saved = save_parquet(additional, "ruonia", tmp_path)
    result = pd.read_parquet(saved)
    assert len(result) == 3
    assert result["ruonia_pct"].tolist() == [14.5, 14.4, 14.3]
    assert result["volume_bln_rub"].isna().sum() == 2
