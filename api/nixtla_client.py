"""Nixtla TimeGPT client wrapper with helper utilities."""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from nixtla import NixtlaClient


NIXTLA_API_KEY = os.getenv(
    "NIXTLA_API_KEY",
    "nixak-9c61a691b619e36d3a10509de0b83d84762b7385fa82ae5baf76dd018ae54648ccf902cb1652397e",
)


@lru_cache(maxsize=1)
def get_client() -> NixtlaClient:
    """Return a singleton NixtlaClient."""
    return NixtlaClient(api_key=NIXTLA_API_KEY)


def rows_to_dataframe(
    rows: list[dict],
    time_col: str = "timestamp",
    target_col: str = "value",
) -> pd.DataFrame:
    """Convert list-of-dicts into TimeGPT-compatible DataFrame.

    TimeGPT expects columns named ``ds`` (datetime) and ``y`` (target).
    If the data contains a ``unique_id`` column it is preserved for
    multi-series support.
    """
    df = pd.DataFrame([r.model_dump() if hasattr(r, "model_dump") else r for r in rows])

    # Rename to standard names expected by Nixtla
    rename_map: dict[str, str] = {}
    if time_col in df.columns and time_col != "ds":
        rename_map[time_col] = "ds"
    if "timestamp" in df.columns and "ds" not in df.columns:
        rename_map["timestamp"] = "ds"
    if target_col in df.columns and target_col != "y":
        rename_map[target_col] = "y"
    if "value" in df.columns and "y" not in df.columns:
        rename_map["value"] = "y"

    df.rename(columns=rename_map, inplace=True)

    # Ensure datetime type
    df["ds"] = pd.to_datetime(df["ds"])

    # Ensure unique_id exists
    if "unique_id" not in df.columns:
        df["unique_id"] = "series_1"

    # Sort by id + time
    df.sort_values(["unique_id", "ds"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def validate_nixtla_connection() -> bool:
    """Quick check that the API key is valid."""
    try:
        client = get_client()
        return client.validate_api_key()
    except Exception:
        return False
