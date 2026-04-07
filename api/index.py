"""FireIO -- Time Series Forecasting & Analytics API powered by Nixtla TimeGPT.

Endpoints
---------
POST /forecast        -- Generate forecasts with confidence intervals
POST /anomaly-detect  -- Detect anomalies in historical data
POST /monitor         -- Real-time monitoring (online anomaly detection)
POST /analytics       -- Summary statistics, trend & seasonality analysis
GET  /health          -- Health check & API key validation
"""

from __future__ import annotations

import traceback
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import (
    AnalyticsRequest,
    AnalyticsResponse,
    AnomalyRequest,
    AnomalyResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    MonitoringRequest,
    MonitoringResponse,
)
from api.nixtla_client import get_client, rows_to_dataframe, validate_nixtla_connection
from api.plotting import analytics_plot, anomaly_plot, forecast_plot, monitoring_plot

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FireIO",
    description=(
        "End-to-end time-series forecasting, anomaly detection and real-time "
        "monitoring API powered by Nixtla TimeGPT."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Return service health and validate Nixtla API key."""
    nixtla_ok = validate_nixtla_connection()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        nixtla_status="connected" if nixtla_ok else "invalid_key",
    )


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

@app.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
def forecast(req: ForecastRequest):
    """Generate time-series forecasts with confidence intervals.

    Accepts any time-series data containing datetime + value columns.
    Returns point forecasts, confidence-interval bounds and a Plotly chart.
    """
    try:
        df = rows_to_dataframe(req.data, req.time_col, req.target_col)
        client = get_client()

        kwargs: dict[str, Any] = dict(
            df=df,
            h=req.horizon,
            level=req.level,
            model=req.model,
            time_col="ds",
            target_col="y",
        )
        if req.finetune_steps > 0:
            kwargs["finetune_steps"] = req.finetune_steps
        if req.freq:
            kwargs["freq"] = req.freq

        forecast_df = client.forecast(**kwargs)

        # Analytics summary
        analytics = _forecast_analytics(df, forecast_df, req.level)

        # Plot
        plot = forecast_plot(df, forecast_df, levels=req.level)

        return ForecastResponse(
            forecast=forecast_df.to_dict(orient="records"),
            plot_url=None,  # Plotly JSON returned inline
            analytics=analytics,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast error: {exc}\n{traceback.format_exc()}")


def _forecast_analytics(
    historical: pd.DataFrame,
    forecast_df: pd.DataFrame,
    levels: list[int],
) -> dict[str, Any]:
    """Compute analytics for the forecast."""
    fc_col = "TimeGPT" if "TimeGPT" in forecast_df.columns else forecast_df.columns[1]
    fc_values = forecast_df[fc_col]

    analytics: dict[str, Any] = {
        "historical_points": len(historical),
        "forecast_points": len(forecast_df),
        "historical_mean": float(historical["y"].mean()),
        "historical_std": float(historical["y"].std()),
        "historical_min": float(historical["y"].min()),
        "historical_max": float(historical["y"].max()),
        "forecast_mean": float(fc_values.mean()),
        "forecast_min": float(fc_values.min()),
        "forecast_max": float(fc_values.max()),
        "forecast_trend": "upward" if float(fc_values.iloc[-1]) > float(fc_values.iloc[0]) else "downward",
    }

    # CI widths
    for lvl in levels:
        lo = f"TimeGPT-lo-{lvl}"
        hi = f"TimeGPT-hi-{lvl}"
        if lo in forecast_df.columns and hi in forecast_df.columns:
            width = (forecast_df[hi] - forecast_df[lo]).mean()
            analytics[f"avg_ci_width_{lvl}"] = float(width)

    return analytics


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------

@app.post("/anomaly-detect", response_model=AnomalyResponse, tags=["Anomaly Detection"])
def anomaly_detect(req: AnomalyRequest):
    """Detect anomalies in historical time-series data.

    Uses TimeGPT to identify data points that fall outside expected
    confidence bounds.
    """
    try:
        df = rows_to_dataframe(req.data, req.time_col, req.target_col)
        client = get_client()

        kwargs: dict[str, Any] = dict(
            df=df,
            time_col="ds",
            target_col="y",
        )
        if req.freq:
            kwargs["freq"] = req.freq

        anomaly_df = client.detect_anomalies(
            **kwargs,
        )

        total = int(anomaly_df["anomaly"].sum()) if "anomaly" in anomaly_df.columns else 0
        ratio = total / len(anomaly_df) if len(anomaly_df) > 0 else 0.0

        analytics = {
            "total_points": len(anomaly_df),
            "total_anomalies": total,
            "anomaly_ratio": round(ratio, 4),
            "mean_value": float(df["y"].mean()),
            "std_value": float(df["y"].std()),
        }

        plot = anomaly_plot(df, anomaly_df)

        return AnomalyResponse(
            anomalies=anomaly_df.to_dict(orient="records"),
            total_anomalies=total,
            anomaly_ratio=round(ratio, 4),
            plot_url=None,
            analytics=analytics,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Anomaly detection error: {exc}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Real-Time Monitoring (Online Detection)
# ---------------------------------------------------------------------------

@app.post("/monitor", response_model=MonitoringResponse, tags=["Real-Time Monitoring"])
def monitor(req: MonitoringRequest):
    """Real-time anomaly monitoring.

    Compares new incoming data points against a forecast generated from
    historical data to flag any that breach confidence bounds.
    """
    try:
        hist_df = rows_to_dataframe(req.data, req.time_col, req.target_col)
        new_df = rows_to_dataframe(req.new_data, req.time_col, req.target_col)

        client = get_client()
        horizon = len(new_df)

        kwargs: dict[str, Any] = dict(
            df=hist_df,
            h=horizon,
            level=req.level,
            time_col="ds",
            target_col="y",
        )
        if req.freq:
            kwargs["freq"] = req.freq

        fc = client.forecast(**kwargs)

        # Compare forecast bounds with actuals
        alerts: list[dict[str, Any]] = []
        fc_col = "TimeGPT" if "TimeGPT" in fc.columns else fc.columns[1]
        lvl = req.level[0] if req.level else 95
        lo_col = f"TimeGPT-lo-{lvl}"
        hi_col = f"TimeGPT-hi-{lvl}"

        for i in range(min(len(fc), len(new_df))):
            actual = float(new_df.iloc[i]["y"])
            expected = float(fc.iloc[i][fc_col])
            lo = float(fc.iloc[i][lo_col]) if lo_col in fc.columns else expected
            hi = float(fc.iloc[i][hi_col]) if hi_col in fc.columns else expected

            if actual < lo or actual > hi:
                alerts.append({
                    "timestamp": str(new_df.iloc[i]["ds"]),
                    "actual_value": actual,
                    "expected_value": expected,
                    "lower_bound": lo,
                    "upper_bound": hi,
                    "deviation": round(actual - expected, 4),
                    "severity": "high" if abs(actual - expected) > 2 * (hi - lo) else "medium",
                })

        plot = monitoring_plot(hist_df, new_df, alerts)

        return MonitoringResponse(
            alerts=alerts,
            total_alerts=len(alerts),
            plot_url=None,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Monitoring error: {exc}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@app.post("/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
def analytics(req: AnalyticsRequest):
    """Comprehensive time-series analytics.

    Returns summary statistics, trend analysis, seasonality estimation
    and a Plotly chart.
    """
    try:
        df = rows_to_dataframe(req.data, req.time_col, req.target_col)
        y = df["y"]

        # Summary
        summary = {
            "count": int(len(y)),
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
            "median": float(y.median()),
            "q25": float(y.quantile(0.25)),
            "q75": float(y.quantile(0.75)),
            "skewness": float(y.skew()),
            "kurtosis": float(y.kurtosis()),
            "range": float(y.max() - y.min()),
            "cv": float(y.std() / y.mean()) if y.mean() != 0 else None,
        }

        # Trend (rolling mean)
        window = max(3, len(df) // 10)
        rolling = y.rolling(window=window, center=True).mean()
        trend_direction = "upward" if float(rolling.dropna().iloc[-1]) > float(rolling.dropna().iloc[0]) else "downward"

        trend = {
            "direction": trend_direction,
            "rolling_window": window,
            "start_level": float(rolling.dropna().iloc[0]),
            "end_level": float(rolling.dropna().iloc[-1]),
            "change_pct": round(
                (float(rolling.dropna().iloc[-1]) - float(rolling.dropna().iloc[0]))
                / abs(float(rolling.dropna().iloc[0]))
                * 100,
                2,
            )
            if float(rolling.dropna().iloc[0]) != 0
            else 0,
        }

        # Simple seasonality via autocorrelation
        seasonality = _estimate_seasonality(y)

        plot = analytics_plot(df, trend=rolling, title="Time Series Analytics")

        return AnalyticsResponse(
            summary=summary,
            seasonality=seasonality,
            trend=trend,
            plot_url=None,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analytics error: {exc}\n{traceback.format_exc()}")


def _estimate_seasonality(y: pd.Series) -> dict[str, Any]:
    """Estimate seasonality via autocorrelation peaks."""
    n = len(y)
    if n < 10:
        return {"detected": False, "reason": "too few data points"}

    # Normalize
    y_norm = (y - y.mean()) / (y.std() + 1e-9)
    max_lag = min(n // 2, 200)

    acf_values = []
    for lag in range(1, max_lag + 1):
        c = float(np.corrcoef(y_norm.iloc[:-lag], y_norm.iloc[lag:])[0, 1])
        acf_values.append(c)

    # Find peak autocorrelation lag (skip lag 0)
    if not acf_values:
        return {"detected": False}

    best_lag = int(np.argmax(acf_values)) + 1
    best_acf = float(acf_values[best_lag - 1])

    return {
        "detected": best_acf > 0.3,
        "estimated_period": best_lag,
        "autocorrelation": round(best_acf, 4),
        "confidence": "high" if best_acf > 0.6 else "medium" if best_acf > 0.3 else "low",
    }


# ---------------------------------------------------------------------------
# CSV Upload endpoint (convenience)
# ---------------------------------------------------------------------------

@app.post("/upload-csv", tags=["Utilities"])
async def upload_csv(file: bytes = None):
    """Parse an uploaded CSV and return it as JSON rows for use with other endpoints."""
    from fastapi import File, UploadFile
    # This is a placeholder -- Vercel serverless may not support large uploads
    return {"message": "Use the JSON endpoints directly for serverless deployments."}


# ---------------------------------------------------------------------------
# Plotly chart endpoint (returns full Plotly JSON)
# ---------------------------------------------------------------------------

@app.post("/forecast/plot", tags=["Forecasting"])
def forecast_with_plot(req: ForecastRequest):
    """Same as /forecast but returns the Plotly chart JSON in the response."""
    result = forecast(req)
    df = rows_to_dataframe(req.data, req.time_col, req.target_col)
    fc_df = pd.DataFrame(result.forecast)
    fc_df["ds"] = pd.to_datetime(fc_df["ds"])
    chart = forecast_plot(df, fc_df, levels=req.level)
    return {"forecast": result.forecast, "analytics": result.analytics, "chart": chart}


@app.post("/anomaly-detect/plot", tags=["Anomaly Detection"])
def anomaly_with_plot(req: AnomalyRequest):
    """Same as /anomaly-detect but returns the Plotly chart JSON."""
    result = anomaly_detect(req)
    df = rows_to_dataframe(req.data, req.time_col, req.target_col)
    anom_df = pd.DataFrame(result.anomalies)
    if "ds" in anom_df.columns:
        anom_df["ds"] = pd.to_datetime(anom_df["ds"])
    chart = anomaly_plot(df, anom_df)
    return {"anomalies": result.anomalies, "analytics": result.analytics, "chart": chart}
