"""Pydantic models for request/response schemas."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class TimeSeriesRow(BaseModel):
    """Single row of time-series data."""
    timestamp: str = Field(..., description="ISO-8601 datetime string")
    value: float = Field(..., description="Target value to forecast / analyse")
    unique_id: Optional[str] = Field("series_1", description="Series identifier for multi-series data")


class TimeSeriesPayload(BaseModel):
    """Generic payload accepted by all endpoints."""
    data: list[TimeSeriesRow] = Field(..., min_length=2, description="Time-series rows")
    freq: Optional[str] = Field(None, description="Pandas frequency string, e.g. 'h', 'D', 'MS'. Auto-detected if omitted.")
    time_col: str = Field("timestamp", description="Name of the datetime column")
    target_col: str = Field("value", description="Name of the target column")


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

class ForecastRequest(TimeSeriesPayload):
    horizon: int = Field(12, ge=1, description="Number of steps to forecast")
    level: list[int] = Field(default=[80, 95], description="Confidence-interval levels (percent)")
    finetune_steps: int = Field(0, ge=0, description="Fine-tune steps for TimeGPT")
    model: str = Field("timegpt-1", description="Model to use: timegpt-1 or timegpt-1-long-horizon")


class ForecastResponse(BaseModel):
    forecast: list[dict[str, Any]]
    plot_url: Optional[str] = None
    analytics: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------

class AnomalyRequest(TimeSeriesPayload):
    level: list[int] = Field(default=[95], description="Confidence level for anomaly bounds")


class AnomalyResponse(BaseModel):
    anomalies: list[dict[str, Any]]
    total_anomalies: int
    anomaly_ratio: float
    plot_url: Optional[str] = None
    analytics: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Real-Time Monitoring (Online Detection)
# ---------------------------------------------------------------------------

class MonitoringRequest(TimeSeriesPayload):
    new_data: list[TimeSeriesRow] = Field(..., min_length=1, description="New incoming data points to check")
    level: list[int] = Field(default=[95], description="Confidence level")


class MonitoringResponse(BaseModel):
    alerts: list[dict[str, Any]]
    total_alerts: int
    plot_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

class AnalyticsRequest(TimeSeriesPayload):
    pass


class AnalyticsResponse(BaseModel):
    summary: dict[str, Any]
    seasonality: dict[str, Any]
    trend: dict[str, Any]
    plot_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    nixtla_status: str
