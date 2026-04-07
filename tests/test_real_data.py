"""Test the deployed FireIO API with the user's real solar/weather CSV dataset.

Converts CSV rows into the API's JSON format:
  - "Date avec l'Heure" -> timestamp
  - "Y1t" -> value (target)
  - All other numeric columns -> features (exogenous variables)

Generates 200+ rows by repeating the 1-day pattern across 4+ days
to meet Nixtla's 144-observation minimum for 30-min frequency.
"""
import csv
import io
import json
import sys
from datetime import datetime, timedelta

import requests

API_BASE = "https://fireio.vercel.app"

# --- Raw CSV data (1 full day = 48 half-hour intervals) ----
CSV_RAW = r"""Date avec l'Heure,Angle du vent,Vitesse vent,Humidite ambiante,Temperature ambiante,Irradiation,Carbon monoxide,Dust,Humidity,Nitrogen dioxide,Ozone,PM10,PM2.5,Pressure,Sulphur dioxide,Temperature,VOC,Visibility,Weather Code,Wind Bearing,Wind Gust,Wind Speed,alder_pollen,birch_pollen,grass_pollen,mugwort_pollen,olive_pollen,ragweed_pollen,Y1t
2022-01-01 00:30:00,178.0,0.4,80.5,6.2,0.0,186.79,16.0,82.98,9.78,42.16,11.0,8.0,1020.0,1.75,7.82,143.83,14.4,3.0,321.0,18.1,7.27,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 01:00:00,178.0,0.3,81.3,5.9,0.0,186.79,16.0,83.42,9.97,41.41,11.0,8.0,1020.0,1.78,7.46,143.83,14.4,3.0,321.0,18.1,7.27,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 01:30:00,180.0,0.1,82.0,5.6,0.0,186.79,16.0,83.87,10.15,40.65,11.0,8.0,1020.0,1.8,7.11,143.83,14.4,3.0,320.0,15.12,5.95,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 02:00:00,316.0,0.1,82.8,5.3,0.0,186.79,16.0,84.31,10.34,39.9,11.0,8.0,1020.0,1.83,6.76,143.83,14.4,3.0,320.0,15.12,5.95,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 02:30:00,316.0,0.2,83.6,5.0,0.0,186.79,16.0,84.31,10.34,39.9,11.0,8.0,1020.0,1.83,6.76,143.83,14.4,3.0,320.0,15.12,5.95,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 03:00:00,328.0,0.3,84.3,4.7,0.0,186.79,16.0,84.76,10.52,39.14,11.0,8.0,1020.0,1.85,6.41,143.83,14.4,3.0,352.0,13.68,5.47,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 03:30:00,338.0,0.5,84.3,4.5,0.0,186.79,16.0,84.76,10.52,39.14,11.0,8.0,1020.0,1.85,6.41,143.83,14.4,3.0,352.0,13.68,5.47,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 04:00:00,345.0,0.3,84.4,4.2,0.0,186.79,16.0,85.2,10.71,38.39,11.0,8.0,1020.0,1.87,6.06,143.83,14.4,3.0,357.0,12.96,4.72,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 04:30:00,349.0,0.4,84.4,4.0,0.0,186.79,16.0,85.2,10.71,38.39,11.0,8.0,1020.0,1.87,6.06,143.83,14.4,3.0,357.0,12.96,4.72,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 05:00:00,359.0,0.2,84.4,3.8,0.0,186.79,16.0,85.65,10.89,37.64,11.0,8.0,1020.0,1.9,5.71,143.83,14.4,3.0,4.0,10.08,3.78,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 05:30:00,6.0,0.2,84.5,3.6,0.0,186.79,16.0,85.65,10.89,37.64,11.0,8.0,1020.0,1.9,5.71,143.83,14.4,3.0,4.0,10.08,3.78,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 06:00:00,5.0,0.3,84.5,3.4,0.0,186.79,16.0,86.09,11.08,36.88,11.0,8.0,1020.0,1.92,5.36,143.83,14.4,3.0,357.0,11.88,4.43,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 06:30:00,358.0,0.5,84.6,3.2,0.0,186.79,16.0,86.09,11.08,36.88,11.0,8.0,1020.0,1.92,5.36,143.83,14.4,3.0,357.0,11.88,4.43,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 07:00:00,352.0,0.5,84.6,3.0,0.0,186.79,16.0,86.54,11.27,36.13,11.0,8.0,1020.0,1.95,5.01,143.83,14.4,3.0,350.0,12.24,4.72,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 07:30:00,350.0,0.6,84.6,2.8,0.0,186.79,16.0,86.54,11.27,36.13,11.0,8.0,1020.0,1.95,5.01,143.83,14.4,3.0,350.0,12.24,4.72,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 08:00:00,346.0,0.3,84.6,2.6,0.0,186.79,16.0,86.98,11.45,35.37,11.0,8.0,1020.0,1.97,4.66,143.83,14.4,3.0,347.0,12.96,5.04,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 08:30:00,345.0,0.7,83.2,2.7,33.0,186.79,16.0,86.98,11.45,35.37,11.0,8.0,1020.0,1.97,4.66,143.83,14.4,3.0,347.0,12.96,5.04,0.0,0.0,0.0,0.0,0.0,0.0,0.37
2022-01-01 09:00:00,346.0,0.6,81.9,2.7,67.0,195.63,16.0,86.39,13.71,34.94,14.0,10.0,1020.0,2.4,4.89,149.02,14.3,3.0,344.0,14.76,5.04,0.0,0.0,0.0,0.0,0.0,0.0,0.81
2022-01-01 09:30:00,345.0,0.3,80.5,2.7,100.0,195.63,16.0,86.39,13.71,34.94,14.0,10.0,1020.0,2.4,4.89,149.02,14.3,3.0,344.0,14.76,5.04,0.0,0.0,0.0,0.0,0.0,0.0,1.22
2022-01-01 10:00:00,341.0,0.2,79.2,2.7,133.0,195.63,16.0,85.8,15.97,34.5,17.0,12.0,1020.0,2.84,5.12,154.21,14.2,3.0,337.0,15.12,5.4,0.0,0.0,0.0,0.0,0.0,0.0,1.6
2022-01-01 10:30:00,354.0,0.5,78.0,2.8,166.0,195.63,16.0,85.8,15.97,34.5,17.0,12.0,1020.0,2.84,5.12,154.21,14.2,3.0,337.0,15.12,5.4,0.0,0.0,0.0,0.0,0.0,0.0,1.91
2022-01-01 11:00:00,351.0,0.5,76.6,2.8,199.0,195.63,16.0,85.21,18.23,34.07,21.0,15.0,1020.0,3.27,5.35,159.4,14.0,3.0,329.0,16.92,5.76,0.0,0.0,0.0,0.0,0.0,0.0,2.26
2022-01-01 11:30:00,343.0,0.7,75.3,2.8,232.0,195.63,16.0,85.21,18.23,34.07,21.0,15.0,1020.0,3.27,5.35,159.4,14.0,3.0,329.0,16.92,5.76,0.0,0.0,0.0,0.0,0.0,0.0,2.62
2022-01-01 12:00:00,349.0,0.4,73.9,2.8,266.0,204.46,16.0,84.62,20.49,33.63,24.0,17.0,1020.0,3.71,5.58,164.59,13.9,3.0,318.0,18.0,6.37,0.0,0.0,0.0,0.0,0.0,0.0,2.93
2022-01-01 12:30:00,336.0,0.7,72.6,2.9,275.0,204.46,16.0,84.62,20.49,33.63,24.0,17.0,1020.0,3.71,5.58,164.59,13.9,3.0,318.0,18.0,6.37,0.0,0.0,0.0,0.0,0.0,0.0,2.87
2022-01-01 13:00:00,330.0,0.5,71.2,2.9,264.0,204.46,16.0,84.03,22.75,33.19,27.0,19.0,1020.0,4.14,5.81,169.79,13.8,3.0,310.0,18.36,6.37,0.0,0.0,0.0,0.0,0.0,0.0,2.76
2022-01-01 13:30:00,308.0,0.5,70.0,3.1,238.0,204.46,16.0,84.03,22.75,33.19,27.0,19.0,1020.0,4.14,5.81,169.79,13.8,3.0,310.0,18.36,6.37,0.0,0.0,0.0,0.0,0.0,0.0,2.55
2022-01-01 14:00:00,296.0,0.2,69.2,3.1,200.0,213.3,16.0,83.44,25.01,32.76,30.0,22.0,1019.0,4.57,6.04,174.98,13.6,3.0,302.0,18.72,7.34,0.0,0.0,0.0,0.0,0.0,0.0,2.22
2022-01-01 14:30:00,274.0,0.4,68.7,3.0,156.0,213.3,16.0,83.44,25.01,32.76,30.0,22.0,1019.0,4.57,6.04,174.98,13.6,3.0,302.0,18.72,7.34,0.0,0.0,0.0,0.0,0.0,0.0,1.76
2022-01-01 15:00:00,277.0,0.6,68.4,2.7,100.0,213.3,16.0,82.85,27.27,32.32,33.0,24.0,1019.0,5.01,6.27,180.17,13.5,3.0,291.0,20.16,8.24,0.0,0.0,0.0,0.0,0.0,0.0,1.22
2022-01-01 15:30:00,274.0,0.5,68.5,2.3,44.0,213.3,16.0,82.85,27.27,32.32,33.0,24.0,1019.0,5.01,6.27,180.17,13.5,3.0,291.0,20.16,8.24,0.0,0.0,0.0,0.0,0.0,0.0,0.56
2022-01-01 16:00:00,275.0,0.5,69.7,1.9,11.0,222.14,16.0,82.26,29.52,31.89,36.0,26.0,1019.0,5.44,6.5,185.36,13.4,3.0,281.0,21.96,9.5,0.0,0.0,0.0,0.0,0.0,0.0,0.1
2022-01-01 16:30:00,278.0,0.4,70.8,1.5,0.0,222.14,16.0,82.26,29.52,31.89,36.0,26.0,1019.0,5.44,6.5,185.36,13.4,3.0,281.0,21.96,9.5,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 17:00:00,285.0,0.2,72.0,1.1,0.0,222.14,16.0,81.67,31.78,31.45,39.0,28.0,1019.0,5.88,6.73,190.55,13.2,3.0,282.0,23.04,10.48,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 17:30:00,296.0,0.5,73.2,0.7,0.0,222.14,16.0,81.67,31.78,31.45,39.0,28.0,1019.0,5.88,6.73,190.55,13.2,3.0,282.0,23.04,10.48,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 18:00:00,293.0,0.2,74.3,0.3,0.0,230.97,16.0,81.08,34.04,31.01,43.0,31.0,1018.0,6.31,6.96,195.74,13.1,3.0,281.0,22.32,9.72,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 18:30:00,291.0,0.3,75.0,0.1,0.0,230.97,16.0,81.08,34.04,31.01,43.0,31.0,1018.0,6.31,6.96,195.74,13.1,3.0,281.0,22.32,9.72,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 19:00:00,278.0,0.7,75.7,-0.1,0.0,230.97,16.0,80.49,36.3,30.58,46.0,33.0,1018.0,6.75,7.19,200.93,13.0,3.0,274.0,20.88,8.42,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 19:30:00,274.0,0.7,76.3,-0.3,0.0,230.97,16.0,80.49,36.3,30.58,46.0,33.0,1018.0,6.75,7.19,200.93,13.0,3.0,274.0,20.88,8.42,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 20:00:00,278.0,0.4,77.0,-0.4,0.0,239.81,16.0,79.9,38.56,30.14,49.0,35.0,1017.0,7.18,7.42,206.12,12.8,3.0,263.0,19.44,7.56,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 20:30:00,273.0,0.3,77.2,-0.4,0.0,239.81,16.0,79.9,38.56,30.14,49.0,35.0,1017.0,7.18,7.42,206.12,12.8,3.0,263.0,19.44,7.56,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 21:00:00,271.0,0.3,77.4,-0.5,0.0,239.81,16.0,79.3,40.82,29.71,52.0,37.0,1017.0,7.62,7.65,211.32,12.7,3.0,261.0,19.08,7.2,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 21:30:00,271.0,0.4,77.6,-0.5,0.0,239.81,16.0,79.3,40.82,29.71,52.0,37.0,1017.0,7.62,7.65,211.32,12.7,3.0,261.0,19.08,7.2,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 22:00:00,277.0,0.2,77.8,-0.5,0.0,248.65,16.0,78.71,43.08,29.27,55.0,40.0,1017.0,8.05,7.88,216.51,12.6,3.0,262.0,18.0,6.84,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 22:30:00,268.0,0.4,78.1,-0.5,0.0,248.65,16.0,78.71,43.08,29.27,55.0,40.0,1017.0,8.05,7.88,216.51,12.6,3.0,262.0,18.0,6.84,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 23:00:00,278.0,0.2,78.3,-0.5,0.0,248.65,16.0,78.12,45.33,28.84,58.0,42.0,1017.0,8.48,8.12,221.7,12.4,3.0,265.0,17.64,7.27,0.0,0.0,0.0,0.0,0.0,0.0,-0.06
2022-01-01 23:30:00,262.0,0.4,78.5,-0.5,0.0,248.65,16.0,78.12,45.33,28.84,58.0,42.0,1017.0,8.48,8.12,221.7,12.4,3.0,265.0,17.64,7.27,0.0,0.0,0.0,0.0,0.0,0.0,-0.06"""

# Column names
FEATURE_COLS = [
    "Angle du vent", "Vitesse vent", "Humidite ambiante",
    "Temperature ambiante", "Irradiation", "Carbon monoxide",
    "Dust", "Humidity", "Nitrogen dioxide", "Ozone",
    "PM10", "PM2.5", "Pressure", "Sulphur dioxide",
    "Temperature", "VOC", "Visibility", "Weather Code",
    "Wind Bearing", "Wind Gust", "Wind Speed",
    "alder_pollen", "birch_pollen", "grass_pollen",
    "mugwort_pollen", "olive_pollen", "ragweed_pollen",
]

TIMESTAMP_COL = "Date avec l'Heure"
TARGET_COL = "Y1t"


def csv_to_api_rows(csv_text: str):
    """Parse CSV text into list of TimeSeriesRow dicts."""
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    rows = []
    for r in reader:
        features = {}
        for col in FEATURE_COLS:
            if col in r and r[col] not in (None, ""):
                try:
                    features[col] = float(r[col])
                except ValueError:
                    features[col] = r[col]
        rows.append({
            "timestamp": r[TIMESTAMP_COL],
            "value": float(r[TARGET_COL]),
            "features": features,
        })
    return rows


def replicate_to_continuous(base_rows, target_count=200):
    """Replicate the daily pattern into a continuous 30-min time series.

    Generates exactly target_count rows with perfectly spaced 30-min
    intervals starting from 2022-01-01 00:00:00.  Cycles through the
    base data and adds small noise to avoid exact duplicates.
    """
    import random
    random.seed(42)

    n_base = len(base_rows)
    start = datetime(2022, 1, 1, 0, 0, 0)
    all_rows = []

    for i in range(target_count):
        ts = start + timedelta(minutes=30 * i)
        src = base_rows[i % n_base]
        cycle = i // n_base  # which repetition

        noise = 1.0 + random.uniform(-0.05, 0.05) * (cycle > 0)
        new_features = {}
        for k, v in src["features"].items():
            if isinstance(v, (int, float)):
                new_features[k] = round(v * (1.0 + random.uniform(-0.03, 0.03) * (cycle > 0)), 4)
            else:
                new_features[k] = v

        val = src["value"]
        new_val = round(val * noise if val != 0 else val + random.uniform(-0.01, 0.01) * (cycle > 0), 4)

        all_rows.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "value": new_val,
            "features": new_features,
        })
    return all_rows


def test_forecast(rows):
    """Test /forecast endpoint with exogenous features."""
    print("=" * 60)
    print("TEST: POST /forecast (Y1t with 27 exogenous features)")
    print("=" * 60)

    payload = {
        "data": rows,
        "horizon": 12,
        "freq": "30min",
        "level": [80, 95],
        "model": "timegpt-1",
    }

    print(f"  Rows: {len(rows)}")
    print(f"  Features per row: {len(rows[0]['features'])}")
    feature_names = list(rows[0]["features"].keys())
    print(f"  Feature names: {feature_names[:5]}... (+ {len(feature_names) - 5} more)")
    print(f"  Horizon: {payload['horizon']}")
    print()

    resp = requests.post(f"{API_BASE}/forecast", json=payload, timeout=180)
    print(f"  Status: {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        print(f"  Forecast points: {len(data['forecast'])}")
        print(f"  Exogenous features used: {len(data.get('exogenous_features_used', []))} features")
        print(f"  Analytics keys: {list(data.get('analytics', {}).keys())}")
        print()
        for pt in data["forecast"][:3]:
            print(f"    {pt}")
        if len(data["forecast"]) > 3:
            print(f"    ... ({len(data['forecast']) - 3} more)")
    else:
        print(f"  Error: {resp.text[:600]}")
    print()
    return resp.status_code


def test_analytics(rows):
    """Test /analytics endpoint."""
    print("=" * 60)
    print("TEST: POST /analytics (decomposition + feature analysis)")
    print("=" * 60)

    payload = {"data": rows, "freq": "30min"}
    resp = requests.post(f"{API_BASE}/analytics", json=payload, timeout=120)
    print(f"  Status: {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        summary = data.get("summary", {})
        print(f"  Count: {summary.get('count')}, Mean: {summary.get('mean'):.4f}, Std: {summary.get('std'):.4f}")
        print(f"  Min: {summary.get('min')}, Max: {summary.get('max')}, Median: {summary.get('median')}")
        print(f"  Seasonality: {json.dumps(data.get('seasonality', {}))}")
        print(f"  Trend direction: {data.get('trend', {}).get('direction')}")
        fa = data.get("feature_analysis")
        if fa:
            top = fa.get("importance_ranking", [])[:8]
            print(f"  Top 8 features by |correlation| with Y1t:")
            for f in top:
                print(f"    {f['feature']}: |r| = {f.get('abs_correlation', 'N/A')}")
    else:
        print(f"  Error: {resp.text[:500]}")
    print()
    return resp.status_code


def test_forecast_chart(rows):
    """Test /forecast/chart endpoint (HTML response)."""
    print("=" * 60)
    print("TEST: POST /forecast/chart (interactive Plotly HTML)")
    print("=" * 60)

    payload = {
        "data": rows,
        "horizon": 12,
        "freq": "30min",
        "level": [80, 95],
        "model": "timegpt-1",
    }
    resp = requests.post(f"{API_BASE}/forecast/chart", json=payload, timeout=180)
    print(f"  Status: {resp.status_code}")
    print(f"  Content-Type: {resp.headers.get('content-type', 'N/A')}")
    if resp.status_code == 200:
        html = resp.text
        print(f"  HTML size: {len(html):,} bytes")
        print(f"  Contains Plotly: {'plotly' in html.lower()}")
        print(f"  Contains subplots: {'subplot' in html.lower() or 'domain' in html.lower()}")
    else:
        print(f"  Error: {resp.text[:500]}")
    print()
    return resp.status_code


def test_analytics_chart(rows):
    """Test /analytics/chart (HTML with decomposition + feature plots)."""
    print("=" * 60)
    print("TEST: POST /analytics/chart (interactive analytics HTML)")
    print("=" * 60)

    payload = {"data": rows, "freq": "30min"}
    resp = requests.post(f"{API_BASE}/analytics/chart", json=payload, timeout=120)
    print(f"  Status: {resp.status_code}")
    print(f"  Content-Type: {resp.headers.get('content-type', 'N/A')}")
    if resp.status_code == 200:
        html = resp.text
        print(f"  HTML size: {len(html):,} bytes")
        print(f"  Contains Plotly: {'plotly' in html.lower()}")
    else:
        print(f"  Error: {resp.text[:500]}")
    print()
    return resp.status_code


def test_anomaly_detect(rows):
    """Test /anomaly-detect endpoint."""
    print("=" * 60)
    print("TEST: POST /anomaly-detect (anomaly detection)")
    print("=" * 60)

    payload = {"data": rows, "freq": "30min", "level": [95]}
    resp = requests.post(f"{API_BASE}/anomaly-detect", json=payload, timeout=180)
    print(f"  Status: {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        print(f"  Total anomalies: {data['total_anomalies']}")
        print(f"  Anomaly ratio: {data['anomaly_ratio']:.4f}")
    else:
        print(f"  Error: {resp.text[:500]}")
    print()
    return resp.status_code


if __name__ == "__main__":
    print("Converting CSV to API JSON format...")
    base_rows = csv_to_api_rows(CSV_RAW)
    print(f"Base rows from CSV: {len(base_rows)} (1 day)")

    print("Generating continuous 30-min series (200 points) for 144+ minimum...")
    rows = replicate_to_continuous(base_rows, target_count=200)
    print(f"Total rows: {len(rows)}")
    print(f"Features per row: {len(rows[0]['features'])}")
    print(f"Timestamp range: {rows[0]['timestamp']} -> {rows[-1]['timestamp']}")
    print(f"Target (Y1t) range: {min(r['value'] for r in rows):.4f} to {max(r['value'] for r in rows):.4f}")
    print()

    results = {}
    results["analytics"] = test_analytics(rows)
    results["analytics_chart"] = test_analytics_chart(rows)
    results["forecast"] = test_forecast(rows)
    results["forecast_chart"] = test_forecast_chart(rows)
    results["anomaly_detect"] = test_anomaly_detect(rows)

    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, status in results.items():
        ok = "PASS" if status == 200 else "FAIL"
        if status != 200:
            all_pass = False
        print(f"  {name}: {ok} (HTTP {status})")

    print()
    if all_pass:
        print("All tests passed!")
    else:
        print("Some tests failed.")
        sys.exit(1)
