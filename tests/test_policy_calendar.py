"""CBR calendar scraper tests: no live network access required."""

from datetime import date

import pandas as pd
import pytest

from russian_rates.policy_calendar import (
    CbrCalendarFormatError,
    fetch_policy_calendar,
    parse_policy_calendar,
    save_policy_calendar,
)


def _event(day, desc):
    return f"<article><h3>{day}</h3><p>{desc}</p></article>"


REGULAR = "Заседание Совета директоров Банка России по ключевой ставке"
OLD_REGULAR = "Заседание Совета директоров Банка России по денежно-кредитной политике"
EXTRAORDINARY = "Внеочередное заседание Совета директоров Банка России по ключевой ставке"


def test_parse_newer_and_older_calendar_with_future_dates_and_extraordinary():
    html = (
        "<html><body><nav>2027 год 2026 год 2023 год 2019 год 2018 год 2017 год 2016 год</nav>"
        + _event("12 февраля 2027 года", REGULAR)
        + _event("25 февраля 2027 года", "Резюме обсуждения ключевой ставки")
        + _event("18 декабря 2026 года", REGULAR)
        + _event("19 декабря 2025 года", REGULAR)
        + _event("20 декабря 2024 года", REGULAR)
        + _event("15 августа 2023 года", EXTRAORDINARY)
        + _event("16 декабря 2022 года", REGULAR)
        + _event("17 декабря 2021 года", REGULAR)
        + _event("18 декабря 2020 года", OLD_REGULAR)
        + _event("8 февраля", OLD_REGULAR)
        + _event("13 декабря", OLD_REGULAR)
        + _event("9 февраля", OLD_REGULAR)
        + _event("14 декабря", OLD_REGULAR)
        + _event("3 февраля", OLD_REGULAR)
        + _event("15 декабря", OLD_REGULAR)
        + _event("29 января", OLD_REGULAR)
        + _event("16 декабря", OLD_REGULAR)
        + "</body></html>"
    )
    result = parse_policy_calendar(html, start_year=2016, as_of="2026-11-27")
    assert list(result["date"].dt.year) == (
        [2016] * 2 + [2017] * 2 + [2018] * 2 + [2019] * 2
        + [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]
    )
    assert result.loc[result["date"] == pd.Timestamp("2023-08-15"), "meeting_type"].item() == "extraordinary"
    assert result.loc[result["date"] == pd.Timestamp("2027-02-12"), "status"].item() == "scheduled"
    assert result.loc[result["date"] == pd.Timestamp("2023-08-15"), "status"].item() == "past"
    assert pd.Timestamp("2027-02-25") not in set(result["date"])
    assert len(result) == 16


def test_parse_rejects_missing_events():
    with pytest.raises(CbrCalendarFormatError, match="No policy-rate meetings"):
        parse_policy_calendar(
            "<html><p>12 февраля 2027 года</p><p>Резюме обсуждения ключевой ставки</p></html>"
        )


def test_parse_rejects_duplicate_meeting_dates():
    html = "<body>" + _event("12 февраля 2027 года", REGULAR) * 2 + "</body>"
    with pytest.raises(CbrCalendarFormatError, match="Duplicate"):
        parse_policy_calendar(html)


def test_parse_rejects_interior_missing_year():
    html = (
        _event("12 февраля 2027 года", REGULAR)
        + _event("11 февраля 2025 года", REGULAR)
    )
    with pytest.raises(CbrCalendarFormatError, match="missing years"):
        parse_policy_calendar(html)


def test_parse_can_filter_years():
    html = (
        _event("12 февраля 2027 года", REGULAR)
        + _event("18 декабря 2026 года", REGULAR)
        + _event("20 декабря 2024 года", REGULAR)
    )
    got = parse_policy_calendar(html, start_year=2026, end_year=2027, as_of=date(2026, 11, 27))
    assert got["year"].tolist() == [2026, 2027]


class _Response:
    text = _event("12 февраля 2027 года", REGULAR)

    def raise_for_status(self):
        pass


class _Session:
    def __init__(self):
        self.urls = []

    def get(self, url, timeout):
        self.urls.append((url, timeout))
        return _Response()


def test_fetch_uses_explicit_source_and_timeout():
    session = _Session()
    result = fetch_policy_calendar(
        start_year=2027, as_of="2026-11-27", session=session
    )
    assert len(result) == 1
    assert session.urls == [("https://www.cbr.ru/dkp/cal_mp/", 30)]


def test_fetch_rejects_missing_start_year():
    with pytest.raises(CbrCalendarFormatError, match="Requested start year"):
        fetch_policy_calendar(
            start_year=2016, as_of="2026-11-27", session=_Session()
        )


def test_save_and_reload_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    frame = parse_policy_calendar(
        _event("12 февраля 2027 года", REGULAR),
        start_year=2027,
        as_of="2026-11-27",
    )
    path = tmp_path / "calendar.parquet"
    assert save_policy_calendar(frame, path) == path
    loaded = pd.read_parquet(path)
    pd.testing.assert_frame_equal(frame, loaded)
    assert not (tmp_path / ".calendar.parquet.tmp").exists()
