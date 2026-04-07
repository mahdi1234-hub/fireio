"""Pydantic models for request/response schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class TimeSeriesRow(BaseModel):
    """Single row of time-series data with optional exogenous features.

    The ``features`` dict can contain any number of numerical or categorical
    variables that may affect the target value.  For example:

        {"timestamp": "2024-01-01", "value": 100, "features": {"temperature": 22.5, "day_of_week": "Monday", "is_holiday": 1}}
    """
    timestamp: str = Field(..., description="ISO-8601 datetime string")
    value: float = Field(..., description="Target value to forecast / analyse")
    unique_id: Optional[str] = Field("series_1", description="Series identifier for multi-series data")
    features: Optional[Dict[str, Any]] = Field(
        None,
        description="Exogenous variables (numerical or categorical) that affect the target. "
        "E.g. {'temperature': 22.5, 'day_of_week': 'Monday', 'promo': 1}",
    )


class FutureFeatureRow(BaseModel):
    """Exogenous feature values for future time steps (needed for forecast with exogenous vars)."""
    timestamp: str = Field(..., description="ISO-8601 datetime string for the future step")
    features: Dict[str, Any] = Field(..., description="Feature values for this future step")


class TimeSeriesPayload(BaseModel):
    """Generic payload accepted by all endpoints."""
    data: List[TimeSeriesRow] = Field(..., min_length=2, description="Time-series rows")
    freq: Optional[str] = Field(None, description="Pandas frequency string, e.g. 'h', 'D', 'MS'. Auto-detected if omitted.")
    time_col: str = Field("timestamp", description="Name of the datetime column")
    target_col: str = Field("value", description="Name of the target column")


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

class ForecastRequest(TimeSeriesPayload):
    horizon: int = Field(12, ge=1, description="Number of steps to forecast")
    level: List[int] = Field(default=[80, 95], description="Confidence-interval levels (percent)")
    finetune_steps: int = Field(0, ge=0, description="Fine-tune steps for TimeGPT")
    model: str = Field("timegpt-1", description="Model to use: timegpt-1 or timegpt-1-long-horizon")
    future_features: Optional[List[FutureFeatureRow]] = Field(
        None,
        description="Exogenous feature values for each future time step. "
        "Required when historical data includes features and you want exogenous forecasting. "
        "Must have exactly 'horizon' rows.",
    )


class ForecastResponse(BaseModel):
    forecast: List[Dict[str, Any]]
    plot_url: Optional[str] = None
    analytics: Dict[str, Any] = {}
    exogenous_features_used: List[str] = []


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------

class AnomalyRequest(TimeSeriesPayload):
    level: List[int] = Field(default=[95], description="Confidence level for anomaly bounds")


class AnomalyResponse(BaseModel):
    anomalies: List[Dict[str, Any]]
    total_anomalies: int
    anomaly_ratio: float
    plot_url: Optional[str] = None
    analytics: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Real-Time Monitoring (Online Detection)
# ---------------------------------------------------------------------------

class MonitoringRequest(TimeSeriesPayload):
    new_data: List[TimeSeriesRow] = Field(..., min_length=1, description="New incoming data points to check")
    level: List[int] = Field(default=[95], description="Confidence level")


class MonitoringResponse(BaseModel):
    alerts: List[Dict[str, Any]]
    total_alerts: int
    plot_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

class AnalyticsRequest(TimeSeriesPayload):
    pass


class AnalyticsResponse(BaseModel):
    summary: Dict[str, Any]
    seasonality: Dict[str, Any]
    trend: Dict[str, Any]
    feature_analysis: Optional[Dict[str, Any]] = None
    plot_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    nixtla_status: str
