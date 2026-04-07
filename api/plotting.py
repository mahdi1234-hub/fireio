"""Plotting utilities -- rich Plotly charts with full time-series analytics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Forecast chart (enhanced with analytics panels)
# ---------------------------------------------------------------------------

def forecast_plot(
    historical: pd.DataFrame,
    forecast_df: pd.DataFrame,
    levels: Optional[List[int]] = None,
    title: str = "TimeGPT Forecast with Confidence Intervals",
) -> Dict[str, Any]:
    """Multi-panel forecast chart:
    Row 1: Historical + forecast + CI bands
    Row 2: Historical decomposition (trend + residuals)
    Row 3: Distribution / histogram of historical values
    """
    y = historical["y"]
    n = len(y)
    window = max(3, n // 10)

    # Decompose historical
    trend = y.rolling(window=window, center=True).mean()
    detrended = y - trend
    residuals = detrended - detrended.rolling(window=max(3, window // 2), center=True).mean()

    fig = make_subplots(
        rows=3, cols=2,
        row_heights=[0.5, 0.25, 0.25],
        column_widths=[0.7, 0.3],
        subplot_titles=[
            "Forecast with Confidence Intervals", "Forecast Distribution",
            "Trend Component", "Residuals",
            "Seasonal / Cyclical Pattern", "Autocorrelation",
        ],
        specs=[
            [{"colspan": 1}, {"colspan": 1}],
            [{"colspan": 1}, {"colspan": 1}],
            [{"colspan": 1}, {"colspan": 1}],
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    # --- Row 1, Col 1: Main forecast chart ---
    fig.add_trace(
        go.Scatter(
            x=historical["ds"].astype(str).tolist(),
            y=y.tolist(),
            mode="lines",
            name="Historical",
            line=dict(color="#1f77b4", width=1.5),
        ),
        row=1, col=1,
    )

    forecast_col = "TimeGPT" if "TimeGPT" in forecast_df.columns else forecast_df.columns[1]
    fig.add_trace(
        go.Scatter(
            x=forecast_df["ds"].astype(str).tolist(),
            y=forecast_df[forecast_col].tolist(),
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#ff7f0e", width=2),
            marker=dict(size=5),
        ),
        row=1, col=1,
    )

    # CI bands
    ci_colors = [
        ("rgba(255,127,14,0.20)", "rgba(255,127,14,0.35)"),
        ("rgba(255,127,14,0.10)", "rgba(255,127,14,0.20)"),
        ("rgba(255,127,14,0.05)", "rgba(255,127,14,0.10)"),
    ]
    if levels:
        for idx, lvl in enumerate(sorted(levels)):
            lo_col = f"TimeGPT-lo-{lvl}"
            hi_col = f"TimeGPT-hi-{lvl}"
            if lo_col in forecast_df.columns and hi_col in forecast_df.columns:
                fill_c, line_c = ci_colors[idx % len(ci_colors)]
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df["ds"].astype(str).tolist(),
                        y=forecast_df[hi_col].tolist(),
                        mode="lines", line=dict(width=0.5, color=line_c),
                        showlegend=False,
                    ),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df["ds"].astype(str).tolist(),
                        y=forecast_df[lo_col].tolist(),
                        mode="lines", line=dict(width=0.5, color=line_c),
                        fill="tonexty", fillcolor=fill_c,
                        name=f"{lvl}% CI",
                    ),
                    row=1, col=1,
                )

    # --- Row 1, Col 2: Forecast distribution ---
    fc_vals = forecast_df[forecast_col].tolist()
    fig.add_trace(
        go.Histogram(
            y=y.tolist(),
            name="Historical dist",
            marker_color="#1f77b4",
            opacity=0.6,
            nbinsy=20,
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Histogram(
            y=fc_vals,
            name="Forecast dist",
            marker_color="#ff7f0e",
            opacity=0.6,
            nbinsy=10,
        ),
        row=1, col=2,
    )

    # --- Row 2, Col 1: Trend ---
    fig.add_trace(
        go.Scatter(
            x=historical["ds"].astype(str).tolist(),
            y=y.tolist(),
            mode="lines",
            name="Observed",
            line=dict(color="#1f77b4", width=0.8),
            opacity=0.4,
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=historical["ds"].astype(str).tolist(),
            y=trend.tolist(),
            mode="lines",
            name="Trend",
            line=dict(color="#d62728", width=2, dash="solid"),
        ),
        row=2, col=1,
    )

    # --- Row 2, Col 2: Residuals ---
    res_clean = residuals.dropna()
    fig.add_trace(
        go.Scatter(
            x=historical["ds"].iloc[res_clean.index].astype(str).tolist(),
            y=res_clean.tolist(),
            mode="markers+lines",
            name="Residuals",
            line=dict(color="#7f7f7f", width=0.5),
            marker=dict(size=3, color=res_clean.tolist(), colorscale="RdBu", cmin=-res_clean.abs().max(), cmax=res_clean.abs().max()),
        ),
        row=2, col=2,
    )
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)

    # --- Row 3, Col 1: Seasonal / Cyclical pattern ---
    seasonal = detrended.dropna()
    fig.add_trace(
        go.Scatter(
            x=historical["ds"].iloc[seasonal.index].astype(str).tolist(),
            y=seasonal.tolist(),
            mode="lines",
            name="Seasonal+Cycle",
            line=dict(color="#2ca02c", width=1.5),
        ),
        row=3, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

    # --- Row 3, Col 2: Autocorrelation ---
    acf_vals = _compute_acf(y, max_lag=min(40, n // 2))
    fig.add_trace(
        go.Bar(
            x=list(range(1, len(acf_vals) + 1)),
            y=acf_vals,
            name="ACF",
            marker_color=["#1f77b4" if abs(v) > 1.96 / np.sqrt(n) else "#aec7e8" for v in acf_vals],
        ),
        row=3, col=2,
    )
    # Significance bounds
    sig = 1.96 / np.sqrt(n)
    fig.add_hline(y=sig, line_dash="dot", line_color="red", row=3, col=2)
    fig.add_hline(y=-sig, line_dash="dot", line_color="red", row=3, col=2)

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        template="plotly_white",
        hovermode="x unified",
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig.to_dict()


# ---------------------------------------------------------------------------
# Anomaly chart
# ---------------------------------------------------------------------------

def anomaly_plot(
    df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    title: str = "Anomaly Detection",
) -> Dict[str, Any]:
    """Plot with anomaly bounds, flagged points, and residual analysis."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=["Observed vs Expected Bounds", "Deviation from Expected"],
        vertical_spacing=0.1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["ds"].astype(str).tolist(),
            y=df["y"].tolist(),
            mode="lines",
            name="Observed",
            line=dict(color="#1f77b4"),
        ),
        row=1, col=1,
    )

    hi_col, lo_col = _find_bound_cols(anomaly_df)

    if hi_col and lo_col:
        fig.add_trace(
            go.Scatter(
                x=anomaly_df["ds"].astype(str).tolist(),
                y=anomaly_df[hi_col].tolist(),
                mode="lines", line=dict(width=0),
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=anomaly_df["ds"].astype(str).tolist(),
                y=anomaly_df[lo_col].tolist(),
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(44,160,44,0.15)",
                name="Normal Range",
            ),
            row=1, col=1,
        )

    if "anomaly" in anomaly_df.columns:
        anom_points = anomaly_df[anomaly_df["anomaly"] == True]
        if not anom_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=anom_points["ds"].astype(str).tolist(),
                    y=anom_points["y"].tolist(),
                    mode="markers",
                    name="Anomaly",
                    marker=dict(color="red", size=10, symbol="x", line=dict(width=2)),
                ),
                row=1, col=1,
            )

    # Row 2: Deviation
    if "TimeGPT" in anomaly_df.columns and "y" in anomaly_df.columns:
        deviation = anomaly_df["y"] - anomaly_df["TimeGPT"]
        colors = ["red" if anomaly_df.iloc[i].get("anomaly", False) else "#7f7f7f" for i in range(len(anomaly_df))]
        fig.add_trace(
            go.Bar(
                x=anomaly_df["ds"].astype(str).tolist(),
                y=deviation.tolist(),
                name="Deviation",
                marker_color=colors,
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        template="plotly_white",
        hovermode="x unified",
        height=700,
    )
    return fig.to_dict()


# ---------------------------------------------------------------------------
# Monitoring chart
# ---------------------------------------------------------------------------

def monitoring_plot(
    historical: pd.DataFrame,
    new_points: pd.DataFrame,
    alerts: List[Dict],
    title: str = "Real-Time Monitoring",
) -> Dict[str, Any]:
    """Plot historical + new data with alert markers and deviation panel."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=["Time Series with Alerts", "Alert Severity"],
        vertical_spacing=0.1,
    )

    fig.add_trace(
        go.Scatter(
            x=historical["ds"].astype(str).tolist(),
            y=historical["y"].tolist(),
            mode="lines",
            name="Historical",
            line=dict(color="#1f77b4"),
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=new_points["ds"].astype(str).tolist(),
            y=new_points["y"].tolist(),
            mode="lines+markers",
            name="New Data",
            line=dict(color="#2ca02c"),
        ),
        row=1, col=1,
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
                    marker=dict(color="red", size=12, symbol="triangle-up", line=dict(width=2, color="darkred")),
                ),
                row=1, col=1,
            )

            # Row 2: Deviation bars
            fig.add_trace(
                go.Bar(
                    x=alert_df["timestamp"].tolist(),
                    y=alert_df["deviation"].tolist(),
                    name="Deviation",
                    marker_color=["#d62728" if s == "high" else "#ff7f0e" for s in alert_df.get("severity", ["medium"] * len(alert_df))],
                ),
                row=2, col=1,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        template="plotly_white",
        hovermode="x unified",
        height=650,
    )
    return fig.to_dict()


# ---------------------------------------------------------------------------
# Analytics chart (full decomposition)
# ---------------------------------------------------------------------------

def analytics_plot(
    df: pd.DataFrame,
    trend: Optional[pd.Series] = None,
    title: str = "Time Series Analytics",
    exog_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Full analytics dashboard:
    Row 1: Observed + Trend
    Row 2: Seasonal / Cyclical component
    Row 3: Residuals
    Row 4: ACF (autocorrelation)
    Row 5+: Exogenous features (if any)
    """
    y = df["y"]
    n = len(y)
    window = max(3, n // 10)

    # Decompose
    if trend is None:
        trend = y.rolling(window=window, center=True).mean()
    detrended = y - trend
    seasonal_smooth = detrended.rolling(window=max(3, window // 2), center=True).mean()
    residuals = detrended - seasonal_smooth

    n_exog = len(exog_cols) if exog_cols else 0
    n_rows = 4 + n_exog

    subplot_titles = [
        "Observed & Trend",
        "Seasonal / Cyclical Component",
        "Residuals",
        "Autocorrelation (ACF)",
    ] + [f"Feature: {col}" for col in (exog_cols or [])]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=False,
        subplot_titles=subplot_titles,
        vertical_spacing=0.04,
    )

    # Row 1: Observed + Trend
    fig.add_trace(
        go.Scatter(
            x=df["ds"].astype(str).tolist(),
            y=y.tolist(),
            mode="lines",
            name="Observed",
            line=dict(color="#1f77b4", width=1),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["ds"].astype(str).tolist(),
            y=trend.tolist(),
            mode="lines",
            name="Trend",
            line=dict(color="#d62728", width=2.5, dash="solid"),
        ),
        row=1, col=1,
    )

    # Row 2: Seasonal
    seasonal_clean = detrended.dropna()
    fig.add_trace(
        go.Scatter(
            x=df["ds"].iloc[seasonal_clean.index].astype(str).tolist(),
            y=seasonal_clean.tolist(),
            mode="lines",
            name="Seasonal+Cycle",
            line=dict(color="#2ca02c", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(44,160,44,0.1)",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Row 3: Residuals
    res_clean = residuals.dropna()
    fig.add_trace(
        go.Scatter(
            x=df["ds"].iloc[res_clean.index].astype(str).tolist(),
            y=res_clean.tolist(),
            mode="markers+lines",
            name="Residuals",
            marker=dict(
                size=4,
                color=res_clean.tolist(),
                colorscale="RdBu",
                cmin=-res_clean.abs().max() if len(res_clean) > 0 else -1,
                cmax=res_clean.abs().max() if len(res_clean) > 0 else 1,
                showscale=True,
                colorbar=dict(title="Residual", len=0.15, y=0.4),
            ),
            line=dict(color="#7f7f7f", width=0.5),
        ),
        row=3, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

    # +/- 2 std bands for residuals
    if len(res_clean) > 0:
        res_std = float(res_clean.std())
        fig.add_hline(y=2 * res_std, line_dash="dot", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=-2 * res_std, line_dash="dot", line_color="red", opacity=0.5, row=3, col=1)

    # Row 4: ACF
    acf_vals = _compute_acf(y, max_lag=min(40, n // 2))
    sig = 1.96 / np.sqrt(n) if n > 0 else 0
    fig.add_trace(
        go.Bar(
            x=list(range(1, len(acf_vals) + 1)),
            y=acf_vals,
            name="ACF",
            marker_color=["#1f77b4" if abs(v) > sig else "#c7d4e8" for v in acf_vals],
        ),
        row=4, col=1,
    )
    fig.add_hline(y=sig, line_dash="dot", line_color="red", row=4, col=1)
    fig.add_hline(y=-sig, line_dash="dot", line_color="red", row=4, col=1)
    fig.add_hline(y=0, line_color="gray", row=4, col=1)

    # Exogenous feature rows
    feat_colors = ["#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#ff9896"]
    if exog_cols:
        for i, col in enumerate(exog_cols):
            if col in df.columns:
                color = feat_colors[i % len(feat_colors)]
                fig.add_trace(
                    go.Scatter(
                        x=df["ds"].astype(str).tolist(),
                        y=df[col].tolist(),
                        mode="lines+markers",
                        name=col,
                        line=dict(color=color, width=1.5),
                        marker=dict(size=3),
                    ),
                    row=5 + i, col=1,
                )

    height = 300 + 200 * n_rows
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        template="plotly_white",
        hovermode="x unified",
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )

    return fig.to_dict()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_acf(y: pd.Series, max_lag: int = 40) -> List[float]:
    """Compute autocorrelation function values."""
    n = len(y)
    if n < 4:
        return []
    y_norm = (y - y.mean()) / (y.std() + 1e-9)
    acf = []
    for lag in range(1, min(max_lag + 1, n)):
        c = float(np.corrcoef(y_norm.iloc[:-lag], y_norm.iloc[lag:])[0, 1])
        acf.append(round(c, 4))
    return acf


def _find_bound_cols(anomaly_df: pd.DataFrame):
    """Find the hi/lo bound columns in an anomaly DataFrame."""
    if "TimeGPT-hi-99" in anomaly_df.columns:
        return "TimeGPT-hi-99", "TimeGPT-lo-99"
    if "TimeGPT-hi-95" in anomaly_df.columns:
        return "TimeGPT-hi-95", "TimeGPT-lo-95"
    hi_cols = [c for c in anomaly_df.columns if "hi" in c]
    lo_cols = [c for c in anomaly_df.columns if "lo" in c]
    return (hi_cols[0] if hi_cols else None, lo_cols[0] if lo_cols else None)
