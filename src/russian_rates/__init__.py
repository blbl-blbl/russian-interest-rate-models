"""Historical Russian interest-rate data for quantitative research."""

from .cbr import CbrFormatError, fetch_series, parse_date, parse_table, save_parquet

__all__ = ["CbrFormatError", "fetch_series", "parse_date", "parse_table", "save_parquet"]
