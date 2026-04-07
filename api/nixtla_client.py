"""Nixtla TimeGPT client wrapper with helper utilities."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from nixtla import NixtlaClient


NIXTLA_API_KEY = os.getenv(
    "NIXTLA_API_KEY",
    "nixak-b3afb58253e4691bd70c96cc51284a003ad77957623003b5425108894802e55c5fad25138ecd4f8a",
)


@lru_cache(maxsize=1)
def get_client() -> NixtlaClient:
    """Return a singleton NixtlaClient."""
    return NixtlaClient(api_key=NIXTLA_API_KEY)


def rows_to_dataframe(
    rows: list,
    time_col: str = "timestamp",
    target_col: str = "value",
) -> Tuple[pd.DataFrame, List[str]]:
    """Convert list-of-dicts into TimeGPT-compatible DataFrame.

    Returns
    -------
    df : DataFrame with ``ds``, ``y``, ``unique_id`` and any exogenous columns.
    exog_cols : list of exogenous column names extracted from ``features``.
    """
    raw = [r.model_dump() if hasattr(r, "model_dump") else r for r in rows]
    df = pd.DataFrame(raw)

    # Flatten features dict into separate columns
    exog_cols: List[str] = []
    if "features" in df.columns:
        features_series = df["features"]
        non_null = features_series.dropna()
        if len(non_null) > 0:
            feat_df = pd.json_normalize(non_null.tolist())
            feat_df.index = non_null.index
            exog_cols = list(feat_df.columns)

            # Encode categorical features as numeric
            for col in exog_cols:
                if feat_df[col].dtype == object:
                    feat_df[col] = feat_df[col].astype("category").cat.codes

            df = df.join(feat_df)
        df.drop(columns=["features"], inplace=True)

    # Rename to standard names expected by Nixtla
    rename_map: Dict[str, str] = {}
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

    return df, exog_cols


def build_future_exog(
    future_rows: list,
    exog_cols: List[str],
    hist_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Build a future exogenous DataFrame from FutureFeatureRow list.

    If future_rows is None or empty, returns None.
    """
    if not future_rows:
        return None

    raw = [r.model_dump() if hasattr(r, "model_dump") else r for r in future_rows]
    fdf = pd.DataFrame(raw)

    # Flatten features
    if "features" in fdf.columns:
        feat_df = pd.json_normalize(fdf["features"].tolist())
        for col in feat_df.columns:
            if feat_df[col].dtype == object:
                feat_df[col] = feat_df[col].astype("category").cat.codes
        fdf = fdf.drop(columns=["features"]).join(feat_df)

    fdf.rename(columns={"timestamp": "ds"}, inplace=True)
    fdf["ds"] = pd.to_datetime(fdf["ds"])

    # Ensure unique_id
    if "unique_id" not in fdf.columns:
        fdf["unique_id"] = hist_df["unique_id"].iloc[0] if "unique_id" in hist_df.columns else "series_1"

    # Only keep the exog columns + ds + unique_id
    keep = ["ds", "unique_id"] + [c for c in exog_cols if c in fdf.columns]
    fdf = fdf[[c for c in keep if c in fdf.columns]]

    return fdf


def validate_nixtla_connection() -> bool:
    """Quick check that the API key is valid."""
    try:
        client = get_client()
        return client.validate_api_key()
    except Exception:
        return False
