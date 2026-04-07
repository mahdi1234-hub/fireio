"""End-to-end tests for the FireIO API using sample time-series data."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from api.index import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers -- generate synthetic daily data
# ---------------------------------------------------------------------------

def _make_daily_series(n: int = 90, start: str = "2024-01-01") -> list[dict]:
    """Generate a simple sinusoidal time series with some noise."""
    import random
    random.seed(42)
    base = datetime.fromisoformat(start)
    rows = []
    for i in range(n):
        ts = (base + timedelta(days=i)).isoformat()
        val = 100 + 20 * math.sin(2 * math.pi * i / 30) + random.gauss(0, 3)
        rows.append({"timestamp": ts, "value": round(val, 2), "unique_id": "series_1"})
    return rows


SAMPLE_DATA = _make_daily_series(90)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_root(self):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["version"] == "1.0.0"

    def test_health_endpoint(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "nixtla_status" in resp.json()


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

class TestForecast:
    def test_forecast_basic(self):
        payload = {
            "data": SAMPLE_DATA,
            "horizon": 7,
            "level": [80, 95],
            "freq": "D",
        }
        resp = client.post("/forecast", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "forecast" in body
        assert len(body["forecast"]) == 7
        assert "analytics" in body
        assert "historical_mean" in body["analytics"]

    def test_forecast_with_plot(self):
        payload = {
            "data": SAMPLE_DATA,
            "horizon": 5,
            "level": [90],
            "freq": "D",
        }
        resp = client.post("/forecast/plot", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "chart" in body
        assert "data" in body["chart"]  # Plotly figure dict has 'data' key

    def test_forecast_validation_error(self):
        """Empty data should fail validation."""
        payload = {"data": [], "horizon": 5}
        resp = client.post("/forecast", json=payload)
        assert resp.status_code == 422  # Pydantic validation error


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------

class TestAnomalyDetection:
    def test_anomaly_detect(self):
        payload = {
            "data": SAMPLE_DATA,
            "freq": "D",
        }
        resp = client.post("/anomaly-detect", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "anomalies" in body
        assert "total_anomalies" in body
        assert "anomaly_ratio" in body
        assert isinstance(body["anomaly_ratio"], float)

    def test_anomaly_with_plot(self):
        payload = {
            "data": SAMPLE_DATA,
            "freq": "D",
        }
        resp = client.post("/anomaly-detect/plot", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "chart" in body


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

class TestMonitoring:
    def test_monitor(self):
        historical = SAMPLE_DATA[:80]
        new_data = SAMPLE_DATA[80:]
        payload = {
            "data": historical,
            "new_data": new_data,
            "level": [95],
            "freq": "D",
        }
        resp = client.post("/monitor", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "alerts" in body
        assert "total_alerts" in body
        assert isinstance(body["alerts"], list)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

class TestAnalytics:
    def test_analytics(self):
        payload = {
            "data": SAMPLE_DATA,
            "freq": "D",
        }
        resp = client.post("/analytics", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "summary" in body
        assert "seasonality" in body
        assert "trend" in body
        assert body["summary"]["count"] == 90
        assert "mean" in body["summary"]
        assert "std" in body["summary"]
        assert body["trend"]["direction"] in ("upward", "downward")
        assert "detected" in body["seasonality"]
