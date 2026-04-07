"""End-to-end tests for the FireIO API using sample time-series data."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from api.index import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_series(n: int = 90, start: str = "2024-01-01") -> list:
    """Generate a sinusoidal time series with noise and exogenous features."""
    import random
    random.seed(42)
    base = datetime.fromisoformat(start)
    rows = []
    for i in range(n):
        ts = (base + timedelta(days=i)).isoformat()
        temp = 15 + 10 * math.sin(2 * math.pi * i / 365)
        promo = 1 if i % 7 == 5 else 0
        val = round(100 + 20 * math.sin(2 * math.pi * i / 30) + 3 * temp + 15 * promo + random.gauss(0, 3), 2)
        rows.append({
            "timestamp": ts,
            "value": val,
            "unique_id": "series_1",
            "features": {
                "temperature": round(temp, 1),
                "is_promo": promo,
                "day_of_week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i % 7],
            },
        })
    return rows


SAMPLE_DATA = _make_daily_series(90)
SAMPLE_DATA_NO_FEATURES = [
    {"timestamp": r["timestamp"], "value": r["value"], "unique_id": r["unique_id"]}
    for r in SAMPLE_DATA
]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_root(self):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    def test_health_endpoint(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "nixtla_status" in resp.json()


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

class TestForecast:
    def test_forecast_validation_error(self):
        """Empty data should fail validation."""
        payload = {"data": [], "horizon": 5}
        resp = client.post("/forecast", json=payload)
        assert resp.status_code == 422

    def test_forecast_chart_validation_error(self):
        payload = {"data": [], "horizon": 5}
        resp = client.post("/forecast/chart", json=payload)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Analytics (does not call Nixtla API)
# ---------------------------------------------------------------------------

class TestAnalytics:
    def test_analytics_no_features(self):
        payload = {"data": SAMPLE_DATA_NO_FEATURES, "freq": "D"}
        resp = client.post("/analytics", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "summary" in body
        assert "seasonality" in body
        assert "trend" in body
        assert body["summary"]["count"] == 90
        assert body["trend"]["direction"] in ("upward", "downward")

    def test_analytics_with_features(self):
        payload = {"data": SAMPLE_DATA, "freq": "D"}
        resp = client.post("/analytics", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "feature_analysis" in body
        assert body["feature_analysis"] is not None
        assert "features" in body["feature_analysis"]
        assert "temperature" in body["feature_analysis"]["features"]
        assert "is_promo" in body["feature_analysis"]["features"]
        assert "importance_ranking" in body["feature_analysis"]

    def test_analytics_chart(self):
        payload = {"data": SAMPLE_DATA, "freq": "D"}
        resp = client.post("/analytics/chart", json=payload)
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Plotly" in resp.text

    def test_feature_correlation(self):
        """Check that feature correlation is computed correctly."""
        payload = {"data": SAMPLE_DATA, "freq": "D"}
        resp = client.post("/analytics", json=payload)
        body = resp.json()
        fa = body["feature_analysis"]
        temp = fa["features"]["temperature"]
        assert "correlation_with_target" in temp
        assert "correlation_strength" in temp
