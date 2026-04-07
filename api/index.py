"""FireIO -- Time Series Forecasting & Analytics API powered by Nixtla TimeGPT.

Endpoints
---------
POST /forecast           -- Generate forecasts with confidence intervals (JSON)
POST /forecast/chart     -- Forecast with rendered HTML Plotly chart
POST /anomaly-detect     -- Detect anomalies in historical data (JSON)
POST /anomaly-detect/chart -- Anomaly detection with rendered HTML chart
POST /monitor            -- Real-time monitoring / online anomaly detection
POST /monitor/chart      -- Monitoring with rendered HTML chart
POST /analytics          -- Summary statistics, trend & seasonality analysis
POST /analytics/chart    -- Analytics with rendered HTML chart
GET  /health             -- Health check & API key validation

All endpoints support exogenous variables (numerical + categorical features)
that affect the target variable via the ``features`` dict on each data row.
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

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
from api.nixtla_client import (
    build_future_exog,
    get_client,
    rows_to_dataframe,
    validate_nixtla_connection,
)
from api.plotting import analytics_plot, anomaly_plot, forecast_plot, monitoring_plot

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FireIO",
    description=(
        "End-to-end time-series forecasting, anomaly detection and real-time "
        "monitoring API powered by Nixtla TimeGPT. Supports exogenous variables "
        "(numerical and categorical features) that affect the target variable."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: convert Plotly dict to standalone HTML page
# ---------------------------------------------------------------------------

def _plotly_to_html(fig_dict: Dict[str, Any], title: str = "FireIO Chart") -> str:
    """Convert a Plotly figure dict to a self-contained HTML page."""
    import json as _json

    fig_json = _json.dumps(fig_dict, default=str)
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{ margin: 0; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #fafafa; }}
        h1 {{ color: #333; font-size: 1.4rem; margin-bottom: 4px; }}
        .subtitle {{ color: #666; font-size: 0.9rem; margin-bottom: 16px; }}
        #chart {{ width: 100%; height: 600px; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="subtitle">Powered by Nixtla TimeGPT &middot; FireIO API</p>
    <div id="chart"></div>
    <script>
        var figure = {fig_json};
        Plotly.newPlot('chart', figure.data, figure.layout, {{responsive: true}});
    </script>
</body>
</html>"""


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
        version="2.0.0",
        nixtla_status="connected" if nixtla_ok else "invalid_key",
    )


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

@app.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
def forecast(req: ForecastRequest):
    """Generate time-series forecasts with confidence intervals.

    Supports exogenous variables: include a ``features`` dict on each data row
    with numerical or categorical values. For forecasting with exogenous vars,
    also provide ``future_features`` with the same feature columns for each
    future time step.

    Example with features:
    ```json
    {
      "data": [
        {"timestamp": "2024-01-01", "value": 100, "features": {"temperature": 22, "promo": 1}},
        ...
      ],
      "future_features": [
        {"timestamp": "2024-04-01", "features": {"temperature": 25, "promo": 0}},
        ...
      ],
      "horizon": 7, "level": [80, 95], "freq": "D"
    }
    ```
    """
    try:
        df, exog_cols = rows_to_dataframe(req.data, req.time_col, req.target_col)
        client = get_client()

        kwargs: Dict[str, Any] = dict(
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

        # Exogenous variables
        if exog_cols:
            future_df = build_future_exog(req.future_features, exog_cols, df)
            if future_df is not None:
                kwargs["X_df"] = future_df

        forecast_df = client.forecast(**kwargs)

        # Analytics summary
        analytics = _forecast_analytics(df, forecast_df, req.level)

        return ForecastResponse(
            forecast=forecast_df.to_dict(orient="records"),
            plot_url=None,
            analytics=analytics,
            exogenous_features_used=exog_cols,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast error: {exc}\n{traceback.format_exc()}")


@app.post("/forecast/chart", response_class=HTMLResponse, tags=["Forecasting"])
def forecast_chart(req: ForecastRequest):
    """Generate a forecast and return an interactive HTML Plotly chart.

    Open this endpoint in a browser tab to see the rendered chart with
    confidence interval bands.
    """
    try:
        df, exog_cols = rows_to_dataframe(req.data, req.time_col, req.target_col)
        client = get_client()

        kwargs: Dict[str, Any] = dict(
            df=df, h=req.horizon, level=req.level, model=req.model,
            time_col="ds", target_col="y",
        )
        if req.finetune_steps > 0:
            kwargs["finetune_steps"] = req.finetune_steps
        if req.freq:
            kwargs["freq"] = req.freq
        if exog_cols:
            future_df = build_future_exog(req.future_features, exog_cols, df)
            if future_df is not None:
                kwargs["X_df"] = future_df

        forecast_df = client.forecast(**kwargs)
        chart = forecast_plot(df, forecast_df, levels=req.level)

        features_note = f" | Features: {', '.join(exog_cols)}" if exog_cols else ""
        return HTMLResponse(_plotly_to_html(
            chart,
            title=f"Forecast ({req.horizon} steps, CI: {req.level}){features_note}",
        ))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast chart error: {exc}\n{traceback.format_exc()}")


def _forecast_analytics(
    historical: pd.DataFrame,
    forecast_df: pd.DataFrame,
    levels: List[int],
) -> Dict[str, Any]:
    """Compute analytics for the forecast."""
    fc_col = "TimeGPT" if "TimeGPT" in forecast_df.columns else forecast_df.columns[1]
    fc_values = forecast_df[fc_col]

    analytics: Dict[str, Any] = {
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

    Supports exogenous features via the ``features`` dict on each data row.
    """
    try:
        df, exog_cols = rows_to_dataframe(req.data, req.time_col, req.target_col)
        client = get_client()

        kwargs: Dict[str, Any] = dict(df=df, time_col="ds", target_col="y")
        if req.freq:
            kwargs["freq"] = req.freq

        anomaly_df = client.detect_anomalies(**kwargs)

        total = int(anomaly_df["anomaly"].sum()) if "anomaly" in anomaly_df.columns else 0
        ratio = total / len(anomaly_df) if len(anomaly_df) > 0 else 0.0

        analytics: Dict[str, Any] = {
            "total_points": len(anomaly_df),
            "total_anomalies": total,
            "anomaly_ratio": round(ratio, 4),
            "mean_value": float(df["y"].mean()),
            "std_value": float(df["y"].std()),
            "exogenous_features": exog_cols,
        }

        return AnomalyResponse(
            anomalies=anomaly_df.to_dict(orient="records"),
            total_anomalies=total,
            anomaly_ratio=round(ratio, 4),
            plot_url=None,
            analytics=analytics,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Anomaly detection error: {exc}\n{traceback.format_exc()}")


@app.post("/anomaly-detect/chart", response_class=HTMLResponse, tags=["Anomaly Detection"])
def anomaly_chart(req: AnomalyRequest):
    """Detect anomalies and return an interactive HTML Plotly chart."""
    try:
        df, exog_cols = rows_to_dataframe(req.data, req.time_col, req.target_col)
        client = get_client()

        kwargs: Dict[str, Any] = dict(df=df, time_col="ds", target_col="y")
        if req.freq:
            kwargs["freq"] = req.freq

        anomaly_df = client.detect_anomalies(**kwargs)
        chart = anomaly_plot(df, anomaly_df)
        return HTMLResponse(_plotly_to_html(chart, title="Anomaly Detection"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Anomaly chart error: {exc}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Real-Time Monitoring (Online Detection)
# ---------------------------------------------------------------------------

@app.post("/monitor", response_model=MonitoringResponse, tags=["Real-Time Monitoring"])
def monitor(req: MonitoringRequest):
    """Real-time anomaly monitoring.

    Compares new incoming data points against a forecast generated from
    historical data to flag any that breach confidence bounds.
    Supports exogenous features on both historical and new data.
    """
    try:
        hist_df, exog_cols = rows_to_dataframe(req.data, req.time_col, req.target_col)
        new_df, _ = rows_to_dataframe(req.new_data, req.time_col, req.target_col)

        client = get_client()
        horizon = len(new_df)

        kwargs: Dict[str, Any] = dict(
            df=hist_df, h=horizon, level=req.level,
            time_col="ds", target_col="y",
        )
        if req.freq:
            kwargs["freq"] = req.freq

        fc = client.forecast(**kwargs)

        # Compare forecast bounds with actuals
        alerts: List[Dict[str, Any]] = []
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

        return MonitoringResponse(
            alerts=alerts,
            total_alerts=len(alerts),
            plot_url=None,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Monitoring error: {exc}\n{traceback.format_exc()}")


@app.post("/monitor/chart", response_class=HTMLResponse, tags=["Real-Time Monitoring"])
def monitor_chart(req: MonitoringRequest):
    """Real-time monitoring with an interactive HTML Plotly chart."""
    try:
        hist_df, _ = rows_to_dataframe(req.data, req.time_col, req.target_col)
        new_df, _ = rows_to_dataframe(req.new_data, req.time_col, req.target_col)

        client = get_client()
        horizon = len(new_df)
        kwargs: Dict[str, Any] = dict(
            df=hist_df, h=horizon, level=req.level,
            time_col="ds", target_col="y",
        )
        if req.freq:
            kwargs["freq"] = req.freq

        fc = client.forecast(**kwargs)

        # Build alerts
        alerts: List[Dict[str, Any]] = []
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
                })

        chart = monitoring_plot(hist_df, new_df, alerts)
        return HTMLResponse(_plotly_to_html(chart, title=f"Real-Time Monitoring ({len(alerts)} alerts)"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Monitor chart error: {exc}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@app.post("/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
def analytics(req: AnalyticsRequest):
    """Comprehensive time-series analytics including feature analysis.

    When exogenous features are present, returns correlation analysis
    between each feature and the target variable.
    """
    try:
        df, exog_cols = rows_to_dataframe(req.data, req.time_col, req.target_col)
        y = df["y"]

        # Summary
        summary: Dict[str, Any] = {
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

        # Trend
        window = max(3, len(df) // 10)
        rolling = y.rolling(window=window, center=True).mean()
        trend_direction = "upward" if float(rolling.dropna().iloc[-1]) > float(rolling.dropna().iloc[0]) else "downward"

        trend: Dict[str, Any] = {
            "direction": trend_direction,
            "rolling_window": window,
            "start_level": float(rolling.dropna().iloc[0]),
            "end_level": float(rolling.dropna().iloc[-1]),
            "change_pct": round(
                (float(rolling.dropna().iloc[-1]) - float(rolling.dropna().iloc[0]))
                / abs(float(rolling.dropna().iloc[0])) * 100, 2,
            ) if float(rolling.dropna().iloc[0]) != 0 else 0,
        }

        # Seasonality
        seasonality = _estimate_seasonality(y)

        # Feature analysis (correlation with target)
        feature_analysis: Optional[Dict[str, Any]] = None
        if exog_cols:
            feature_analysis = _analyze_features(df, exog_cols)

        return AnalyticsResponse(
            summary=summary,
            seasonality=seasonality,
            trend=trend,
            feature_analysis=feature_analysis,
            plot_url=None,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analytics error: {exc}\n{traceback.format_exc()}")


@app.post("/analytics/chart", response_class=HTMLResponse, tags=["Analytics"])
def analytics_chart(req: AnalyticsRequest):
    """Analytics with an interactive HTML Plotly chart showing trend and features."""
    try:
        df, exog_cols = rows_to_dataframe(req.data, req.time_col, req.target_col)
        y = df["y"]
        window = max(3, len(df) // 10)
        rolling = y.rolling(window=window, center=True).mean()

        chart = analytics_plot(df, trend=rolling, title="Time Series Analytics", exog_cols=exog_cols)
        return HTMLResponse(_plotly_to_html(chart, title="Time Series Analytics"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analytics chart error: {exc}\n{traceback.format_exc()}")


def _estimate_seasonality(y: pd.Series) -> Dict[str, Any]:
    """Estimate seasonality via autocorrelation peaks."""
    n = len(y)
    if n < 10:
        return {"detected": False, "reason": "too few data points"}

    y_norm = (y - y.mean()) / (y.std() + 1e-9)
    max_lag = min(n // 2, 200)

    acf_values = []
    for lag in range(1, max_lag + 1):
        c = float(np.corrcoef(y_norm.iloc[:-lag], y_norm.iloc[lag:])[0, 1])
        acf_values.append(c)

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


def _analyze_features(df: pd.DataFrame, exog_cols: List[str]) -> Dict[str, Any]:
    """Analyse the relationship between exogenous features and the target."""
    result: Dict[str, Any] = {"features": {}}

    for col in exog_cols:
        if col not in df.columns:
            continue

        col_data = df[col]
        feature_info: Dict[str, Any] = {
            "dtype": str(col_data.dtype),
            "unique_values": int(col_data.nunique()),
            "null_count": int(col_data.isna().sum()),
        }

        # Correlation with target (for numeric columns)
        if pd.api.types.is_numeric_dtype(col_data):
            valid = df[["y", col]].dropna()
            if len(valid) > 2:
                corr = float(valid["y"].corr(valid[col]))
                feature_info["correlation_with_target"] = round(corr, 4)
                feature_info["correlation_strength"] = (
                    "strong" if abs(corr) > 0.7
                    else "moderate" if abs(corr) > 0.4
                    else "weak"
                )
                feature_info["mean"] = float(col_data.mean())
                feature_info["std"] = float(col_data.std())
                feature_info["min"] = float(col_data.min())
                feature_info["max"] = float(col_data.max())
        else:
            feature_info["type"] = "categorical"
            feature_info["categories"] = col_data.unique().tolist()[:20]

        result["features"][col] = feature_info

    # Overall feature importance ranking by abs correlation
    ranked = sorted(
        [(k, v.get("correlation_with_target", 0)) for k, v in result["features"].items()],
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    result["importance_ranking"] = [{"feature": k, "abs_correlation": round(abs(v), 4)} for k, v in ranked]

    return result


# ---------------------------------------------------------------------------
# Plotly JSON endpoints (backward compat)
# ---------------------------------------------------------------------------

@app.post("/forecast/plot", tags=["Forecasting"])
def forecast_with_plot(req: ForecastRequest):
    """Same as /forecast but returns the Plotly chart JSON in the response."""
    result = forecast(req)
    df, _ = rows_to_dataframe(req.data, req.time_col, req.target_col)
    fc_df = pd.DataFrame(result.forecast)
    fc_df["ds"] = pd.to_datetime(fc_df["ds"])
    chart = forecast_plot(df, fc_df, levels=req.level)
    return {"forecast": result.forecast, "analytics": result.analytics, "chart": chart}


@app.post("/anomaly-detect/plot", tags=["Anomaly Detection"])
def anomaly_with_plot(req: AnomalyRequest):
    """Same as /anomaly-detect but returns the Plotly chart JSON."""
    result = anomaly_detect(req)
    df, _ = rows_to_dataframe(req.data, req.time_col, req.target_col)
    anom_df = pd.DataFrame(result.anomalies)
    if "ds" in anom_df.columns:
        anom_df["ds"] = pd.to_datetime(anom_df["ds"])
    chart = anomaly_plot(df, anom_df)
    return {"anomalies": result.anomalies, "analytics": result.analytics, "chart": chart}
