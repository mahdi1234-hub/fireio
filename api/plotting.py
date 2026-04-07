"""Plotting utilities -- generate Plotly JSON for confidence-interval charts."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go


def forecast_plot(
    historical: pd.DataFrame,
    forecast_df: pd.DataFrame,
    levels: list[int] | None = None,
    title: str = "TimeGPT Forecast with Confidence Intervals",
) -> dict[str, Any]:
    """Return a Plotly figure as JSON-serialisable dict.

    Parameters
    ----------
    historical : DataFrame with columns ``ds``, ``y``
    forecast_df : DataFrame with ``ds``, ``TimeGPT`` and optional
        ``TimeGPT-lo-XX`` / ``TimeGPT-hi-XX`` columns.
    levels : confidence-interval percentages used during forecasting.
    """
    fig = go.Figure()

    # Historical series
    fig.add_trace(
        go.Scatter(
            x=historical["ds"].astype(str).tolist(),
            y=historical["y"].tolist(),
            mode="lines",
            name="Historical",
            line=dict(color="#1f77b4"),
        )
    )

    # Forecast line
    forecast_col = "TimeGPT" if "TimeGPT" in forecast_df.columns else forecast_df.columns[1]
    fig.add_trace(
        go.Scatter(
            x=forecast_df["ds"].astype(str).tolist(),
            y=forecast_df[forecast_col].tolist(),
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#ff7f0e"),
        )
    )

    # Confidence bands
    colors = ["rgba(255,127,14,0.15)", "rgba(255,127,14,0.08)", "rgba(255,127,14,0.04)"]
    if levels:
        for idx, lvl in enumerate(sorted(levels)):
            lo_col = f"TimeGPT-lo-{lvl}"
            hi_col = f"TimeGPT-hi-{lvl}"
            if lo_col in forecast_df.columns and hi_col in forecast_df.columns:
                color = colors[idx % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df["ds"].astype(str).tolist(),
                        y=forecast_df[hi_col].tolist(),
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df["ds"].astype(str).tolist(),
                        y=forecast_df[lo_col].tolist(),
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor=color,
                        name=f"{lvl}% CI",
                    )
                )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig.to_dict()


def anomaly_plot(
    df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    title: str = "Anomaly Detection",
) -> dict[str, Any]:
    """Plot historical data with anomaly bounds and flagged points."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["ds"].astype(str).tolist(),
            y=df["y"].tolist(),
            mode="lines",
            name="Observed",
            line=dict(color="#1f77b4"),
        )
    )

    # Bounds from anomaly_df
    if "TimeGPT-hi-99" in anomaly_df.columns:
        hi_col = "TimeGPT-hi-99"
        lo_col = "TimeGPT-lo-99"
    elif "TimeGPT-hi-95" in anomaly_df.columns:
        hi_col = "TimeGPT-hi-95"
        lo_col = "TimeGPT-lo-95"
    else:
        # Find any hi/lo columns
        hi_cols = [c for c in anomaly_df.columns if "hi" in c]
        lo_cols = [c for c in anomaly_df.columns if "lo" in c]
        hi_col = hi_cols[0] if hi_cols else None
        lo_col = lo_cols[0] if lo_cols else None

    if hi_col and lo_col:
        fig.add_trace(
            go.Scatter(
                x=anomaly_df["ds"].astype(str).tolist(),
                y=anomaly_df[hi_col].tolist(),
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=anomaly_df["ds"].astype(str).tolist(),
                y=anomaly_df[lo_col].tolist(),
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(44,160,44,0.15)",
                name="Normal Range",
            )
        )

    # Flag anomalies
    if "anomaly" in anomaly_df.columns:
        anom_points = anomaly_df[anomaly_df["anomaly"] == True]
        if not anom_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=anom_points["ds"].astype(str).tolist(),
                    y=anom_points["y"].tolist(),
                    mode="markers",
                    name="Anomaly",
                    marker=dict(color="red", size=10, symbol="x"),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig.to_dict()


def monitoring_plot(
    historical: pd.DataFrame,
    new_points: pd.DataFrame,
    alerts: list[dict],
    title: str = "Real-Time Monitoring",
) -> dict[str, Any]:
    """Plot historical + new data with alert markers."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=historical["ds"].astype(str).tolist(),
            y=historical["y"].tolist(),
            mode="lines",
            name="Historical",
            line=dict(color="#1f77b4"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=new_points["ds"].astype(str).tolist(),
            y=new_points["y"].tolist(),
            mode="lines+markers",
            name="New Data",
            line=dict(color="#2ca02c"),
        )
    )

    if alerts:
        alert_df = pd.DataFrame(alerts)
        if not alert_df.empty and "timestamp" in alert_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=alert_df["timestamp"].tolist(),
                    y=alert_df["actual_value"].tolist(),
                    mode="markers",
                    name="Alert",
                    marker=dict(color="red", size=12, symbol="triangle-up"),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig.to_dict()


def analytics_plot(
    df: pd.DataFrame,
    trend: pd.Series | None = None,
    title: str = "Time Series Analytics",
) -> dict[str, Any]:
    """Plot data with optional trend overlay."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["ds"].astype(str).tolist(),
            y=df["y"].tolist(),
            mode="lines",
            name="Observed",
            line=dict(color="#1f77b4"),
        )
    )

    if trend is not None:
        fig.add_trace(
            go.Scatter(
                x=df["ds"].astype(str).tolist(),
                y=trend.tolist(),
                mode="lines",
                name="Trend (rolling mean)",
                line=dict(color="#d62728", dash="dash"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig.to_dict()
