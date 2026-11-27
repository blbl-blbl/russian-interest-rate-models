"""Bank of Russia policy-rate meeting calendar (regular and extraordinary).

Source: https://www.cbr.ru/dkp/cal_mp/

The calendar includes *meeting dates*, not probabilities or outcomes. Historical
key-rate changes should be aligned to these events separately. Scheduled dates
can be revised; do not treat a published calendar as an immutable forecast.

The CBR page contains a mixture of explicit years (newer entries) and dates
without a year (older calendars, whose sections run newest to oldest).
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .cbr import new_session, parse_date

CALENDAR_URL = "https://www.cbr.ru/dkp/cal_mp/"
COLUMNS = ["date", "year", "meeting_type", "status", "source_url"]

_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
_DATE_LINE = re.compile(
    r"^(?P<day>\d{1,2})\s+"
    r"(?P<month>" + "|".join(_MONTHS) + r")"
    r"(?:\s+(?P<year>20\d{2})\s+года)?(?=$|\s|[|:])",
    flags=re.IGNORECASE,
)
_MEETING = re.compile(
    r"(?:внеочередное\s+)?заседание\s+совета\s+директоров"
    r"[\s\S]{0,170}?(?:ключев(?:ой|ую|ой ставке)|денежно.кредитн)",
    flags=re.IGNORECASE,
)


class CbrCalendarFormatError(ValueError):
    """CBR policy meeting calendar could not be parsed safely."""


def parse_policy_calendar(
    html: str,
    *,
    start_year: int = 2016,
    end_year: int | None = None,
    as_of: date | str | None = None,
) -> pd.DataFrame:
    """Parse CBR's policy meeting dates without inventing missing meetings.

    Older calendar sections (e.g. 2016–2019) omit the year in each row.
    Once an explicit year has been seen, a December-to-January/February
    transition in the page's reverse-year ordering identifies a new year.

    Any site redesign can invalidate that inference. Validate the year counts
    and dated source against the official calendar after downloading.
    """
    if not 2014 <= start_year <= 2099:
        raise ValueError("start_year must be between 2014 and 2099")
    if end_year is not None and end_year < start_year:
        raise ValueError("end_year must be >= start_year")

    cutoff = parse_date(as_of) if as_of is not None else date.today()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()

    # Separation into text lines preserves table rows and section headings.
    raw_lines = soup.get_text(separator="\n", strip=True).splitlines()
    lines = [
        re.sub(r"\s+", " ", line.replace("\xa0", " ").replace("\u202f", " ")).strip()
        for line in raw_lines
    ]
    lines = [line for line in lines if line]

    records: list[dict] = []
    current_year: int | None = None
    previous_month: int | None = None
    event_date: date | None = None
    event_text: list[str] = []

    def finish_event() -> None:
        if event_date is None:
            return
        description = " ".join(event_text)
        if not _MEETING.search(description):
            return
        meeting_type = (
            "extraordinary" if re.search(r"\bвнеочередн", description, re.I)
            else "regular"
        )
        if event_date.year < start_year:
            return
        if end_year is not None and event_date.year > end_year:
            return
        records.append({
            "date": pd.Timestamp(event_date),
            "year": event_date.year,
            "meeting_type": meeting_type,
            "status": "past" if event_date < cutoff else "scheduled",
            "source_url": CALENDAR_URL,
        })

    for line in lines:
        found = _DATE_LINE.match(line)
        if found:
            day = int(found.group("day"))
            month = _MONTHS[found.group("month").lower()]
            explicit_year = found.group("year")

            # Dates with explicit years anchor the descending annual sequence.
            if explicit_year is not None:
                year = int(explicit_year)
            elif current_year is not None:
                year = current_year - int(
                    previous_month is not None and month < previous_month
                )
            else:
                # The navigation contains years, but not dated meetings.
                # Refuse to guess until an explicit dated year is seen.
                continue

            try:
                next_date = date(year, month, day)
            except ValueError as error:
                raise CbrCalendarFormatError(
                    f"Invalid date in CBR calendar: {line!r}"
                ) from error

            finish_event()
            event_date = next_date
            event_text = [line[found.end():]]
            current_year = year
            previous_month = month
        elif event_date is not None:
            event_text.append(line)

    finish_event()

    if not records:
        raise CbrCalendarFormatError(
            "No policy-rate meetings found; check current CBR HTML markup"
        )

    result = (
        pd.DataFrame.from_records(records, columns=COLUMNS)
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    duplicates = result["date"].duplicated(keep=False)
    if duplicates.any():
        dates = result.loc[duplicates, "date"].dt.strftime("%Y-%m-%d").tolist()
        raise CbrCalendarFormatError(f"Duplicate calendar meeting dates: {dates}")

    # Do not silently publish a calendar with whole missing historical years.
    found_years = set(result["year"])
    earliest, latest = min(found_years), max(found_years)
    absent = set(range(earliest, latest + 1)) - found_years
    if absent:
        raise CbrCalendarFormatError(
            f"Calendar contains missing years: {sorted(absent)}"
        )

    return result


def fetch_policy_calendar(
    *,
    start_year: int = 2016,
    end_year: int | None = None,
    as_of: date | str | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download the single public CBR calendar page and parse its meetings."""
    client = session or new_session()
    response = client.get(CALENDAR_URL, timeout=30)
    response.raise_for_status()
    result = parse_policy_calendar(
        response.text, start_year=start_year, end_year=end_year, as_of=as_of
    )
    if start_year not in set(result["year"]):
        raise CbrCalendarFormatError(
            f"Requested start year {start_year} missing from CBR calendar"
        )
    return result


def save_policy_calendar(frame: pd.DataFrame, destination: str | Path) -> Path:
    """Write a Parquet file atomically, including past and future meeting dates."""
    dest = Path(destination)
    if frame.empty or list(frame.columns) != COLUMNS:
        raise ValueError("Expected a nonempty policy meeting calendar with standard columns")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(f".{dest.name}.tmp")
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download CBR policy meeting dates (historical and future)"
    )
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int)
    parser.add_argument(
        "--as-of", help="YYYY-MM-DD; labels meeting dates past/scheduled"
    )
    parser.add_argument(
        "--output", default="data/processed/cbr_policy_meetings.parquet"
    )
    args = parser.parse_args(argv)

    frame = fetch_policy_calendar(
        start_year=args.start_year,
        end_year=args.end_year,
        as_of=args.as_of,
    )
    dest = save_policy_calendar(frame, args.output)
    counts = frame.groupby(["year", "meeting_type"]).size().unstack(fill_value=0)
    print(f"Saved {len(frame)} meetings to {dest}")
    print(counts.to_string())
    print(f"Next scheduled:\n{frame[frame['status'] == 'scheduled'].head(10).to_string(index=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
