"""Download and validate historical interest-rate observations from the Bank of Russia.

Rates are stored in percentage points: 14.5 means 14.5%, not 0.145.
"""

from __future__ import annotations

import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SeriesName = Literal["ruonia", "key_rate", "zcyc"]
SOURCES: dict[str, str] = {
    "ruonia": "https://www.cbr.ru/hd_base/ruonia/dynamics/",
    "key_rate": "https://www.cbr.ru/hd_base/keyrate/",
    "zcyc": "https://www.cbr.ru/hd_base/zcyc_params/",
}
TENORS_YEARS = (0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0)
COLUMNS = {
    "ruonia": ["date", "ruonia_pct", "volume_bln_rub"],
    "key_rate": ["date", "key_rate_pct"],
    "zcyc": ["date", "tenor_years", "yield_pct"],
}
KEYS = {"ruonia": ["date"], "key_rate": ["date"], "zcyc": ["date", "tenor_years"]}
_DATE_RE = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")


class CbrFormatError(ValueError):
    """The CBR page does not have the expected daily observations."""


def parse_date(value: date | str) -> date:
    """Accept ISO YYYY-MM-DD and CBR DD.MM.YYYY dates."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Invalid date {value!r}; use YYYY-MM-DD or DD.MM.YYYY")


def new_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(("GET",)),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({
        "User-Agent": "russian-interest-rate-models/0.1 (public research)",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.5",
    })
    return session


def _year_chunks(start: date, end: date):
    for year in range(start.year, end.year + 1):
        yield max(start, date(year, 1, 1)), min(end, date(year, 12, 31))


def _number(raw: str, *, allow_missing: bool = False) -> float:
    """Parse a CBR number; only explicitly nullable fields may contain a dash."""
    cleaned = raw.replace("\u00a0", "").replace("\u202f", "").replace(" ", "")
    cleaned = cleaned.replace("\u2212", "-").replace(",", ".")
    if allow_missing and cleaned in {"", "-", "–", "—"}:
        return float("nan")
    try:
        result = float(cleaned)
    except ValueError as error:
        raise CbrFormatError(f"Invalid numeric value from CBR: {raw!r}") from error
    if not pd.notna(result) or result == float("inf") or result == float("-inf"):
        raise CbrFormatError(f"Non-finite numeric value from CBR: {raw!r}")
    return result


def parse_table(html: str, series: SeriesName, start: date, end: date) -> pd.DataFrame:
    """Parse date-bearing rows, ignoring multi-level headers."""
    if series not in SOURCES:
        raise ValueError(f"Unknown series: {series}")
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.select(".table-wrapper table")
    if not tables:
        tables = soup.find_all("table")
    if not tables:
        raise CbrFormatError(f"No HTML table found for {series}")

    expected = 13 if series == "zcyc" else 3 if series == "ruonia" else 2
    chosen_rows = None
    for table in tables:
        rows = [
            [cell.get_text(" ", strip=True) for cell in tr.find_all("td", recursive=False)]
            for tr in table.find_all("tr")
        ]
        if any(row and _DATE_RE.fullmatch(row[0]) for row in rows):
            chosen_rows = rows
            break
    if chosen_rows is None:
        # Holiday-only intervals may legitimately contain no observations.
        return pd.DataFrame(columns=COLUMNS[series])

    records = []
    for cells in chosen_rows:
        if not cells or not _DATE_RE.fullmatch(cells[0]):
            continue
        try:
            observed = datetime.strptime(cells[0], "%d.%m.%Y").date()
        except ValueError as error:
            raise CbrFormatError(f"Invalid observation date: {cells[0]!r}") from error
        if not start <= observed <= end:
            continue
        if len(cells) < expected:
            raise CbrFormatError(
                f"{series} on {observed}: expected at least {expected} columns, got {len(cells)}"
            )
        if series == "ruonia":
            records.append((observed, _number(cells[1]), _number(cells[2], allow_missing=True)))
        elif series == "key_rate":
            records.append((observed, _number(cells[1])))
        else:
            records.extend(
                (observed, tenor, _number(raw))
                for tenor, raw in zip(TENORS_YEARS, cells[1:13])
            )

    result = pd.DataFrame.from_records(records, columns=COLUMNS[series])
    if not result.empty:
        result["date"] = pd.to_datetime(result["date"])
        for col in COLUMNS[series][1:]:
            result[col] = result[col].astype("float64")
    return result


def _normalize(frame: pd.DataFrame, series: SeriesName) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"No {series} observations in the requested date range")
    if list(frame.columns) != COLUMNS[series]:
        raise ValueError(f"Unexpected columns for {series}: {list(frame.columns)!r}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    for name in COLUMNS[series][1:]:
        frame[name] = pd.to_numeric(frame[name], errors="raise").astype("float64")
    # RUONIA turnover is auxiliary and may be unpublished (CBR uses an em dash).
    # Interest rates, tenors and dates remain mandatory for model calibration.
    required = [column for column in COLUMNS[series] if column != "volume_bln_rub"]
    if frame[required].isna().any().any():
        raise ValueError(f"Missing data in {series}")
    if frame[KEYS[series]].duplicated().any():
        raise ValueError(f"Duplicate observations in {series}")
    return frame.sort_values(KEYS[series], kind="stable").reset_index(drop=True)


def fetch_series(
    series: SeriesName,
    start: date | str,
    end: date | str,
    *,
    session: requests.Session | None = None,
    pause_seconds: float = 0.2,
) -> pd.DataFrame:
    """Get historical observations in year chunks; respect exact date boundaries."""
    if series not in SOURCES:
        raise ValueError(f"Unknown series: {series}")
    start, end = parse_date(start), parse_date(end)
    if start > end:
        raise ValueError("Start date must not be after end date")
    if pause_seconds < 0:
        raise ValueError("pause_seconds must be non-negative")
    client = session or new_session()
    frames = []
    chunks = list(_year_chunks(start, end))
    for index, (chunk_start, chunk_end) in enumerate(chunks):
        params = {
            "UniDbQuery.Posted": "True",
            "UniDbQuery.From": chunk_start.strftime("%d.%m.%Y"),
            "UniDbQuery.To": chunk_end.strftime("%d.%m.%Y"),
        }
        try:
            response = client.get(SOURCES[series], params=params, timeout=(10, 60))
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(
                f"CBR {series} download failed for {chunk_start}–{chunk_end}: {error}"
            ) from error
        frames.append(parse_table(response.text, series, chunk_start, chunk_end))
        if pause_seconds and index < len(chunks) - 1:
            time.sleep(pause_seconds)
    return _normalize(pd.concat(frames, ignore_index=True), series)


def save_parquet(
    frame: pd.DataFrame,
    series: SeriesName,
    output_dir: str | Path,
    *,
    replace: bool = False,
) -> Path:
    """Upsert observations, preserving other dates; replace Parquet atomically."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{series}.parquet"
    new = _normalize(frame, series)
    if path.exists() and not replace:
        previous = _normalize(pd.read_parquet(path), series)
        combined = pd.concat([previous, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=KEYS[series], keep="last")
        new = _normalize(combined, series)

    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        new.to_parquet(temp_path, engine="pyarrow", index=False)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
    return path
