# FireIO - Time Series Forecasting & Analytics API

End-to-end FastAPI backend powered by **Nixtla TimeGPT** for automated time-series forecasting, anomaly detection and real-time monitoring.

## Features

| Feature | Endpoint | Description |
|---------|----------|-------------|
| **Forecasting** | `POST /forecast` | Generate accurate predictions with confidence intervals |
| **Anomaly Detection** | `POST /anomaly-detect` | Identify unusual patterns in historical data |
| **Real-Time Monitoring** | `POST /monitor` | Detect anomalies as new data arrives |
| **Analytics** | `POST /analytics` | Summary statistics, trend & seasonality analysis |
| **Plots** | `POST /forecast/plot`, `POST /anomaly-detect/plot` | Plotly charts with confidence bands |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn api.index:app --reload

# Run tests
pytest tests/ -v
```

## API Usage

### Forecast

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"timestamp": "2024-01-01", "value": 100},
      {"timestamp": "2024-01-02", "value": 110},
      {"timestamp": "2024-01-03", "value": 105}
    ],
    "horizon": 7,
    "level": [80, 95],
    "freq": "D"
  }'
```

### Anomaly Detection

```bash
curl -X POST http://localhost:8000/anomaly-detect \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"timestamp": "2024-01-01", "value": 100},
      {"timestamp": "2024-01-02", "value": 110},
      {"timestamp": "2024-01-03", "value": 500}
    ],
    "freq": "D"
  }'
```

### Real-Time Monitoring

```bash
curl -X POST http://localhost:8000/monitor \
  -H "Content-Type: application/json" \
  -d '{
    "data": [...historical data...],
    "new_data": [...new incoming points...],
    "level": [95],
    "freq": "D"
  }'
```

### Analytics

```bash
curl -X POST http://localhost:8000/analytics \
  -H "Content-Type: application/json" \
  -d '{
    "data": [...time series data...],
    "freq": "D"
  }'
```

## Data Format

All endpoints accept the same flexible data format:

```json
{
  "data": [
    {"timestamp": "2024-01-01T00:00:00", "value": 42.0, "unique_id": "series_1"},
    {"timestamp": "2024-01-02T00:00:00", "value": 45.0, "unique_id": "series_1"}
  ],
  "freq": "D",
  "time_col": "timestamp",
  "target_col": "value"
}
```

- **timestamp**: ISO-8601 datetime string
- **value**: numeric target to forecast/analyse
- **unique_id**: optional series identifier for multi-series data
- **freq**: Pandas frequency string (D, h, MS, etc.) -- auto-detected if omitted

## Deployment

Deployed on Vercel as a Python serverless function. See `vercel.json` for configuration.

## Tech Stack

- **FastAPI** -- async Python web framework
- **Nixtla TimeGPT** -- foundation model for time-series forecasting
- **Plotly** -- interactive confidence-interval charts
- **Pandas / NumPy** -- data manipulation
- **Pydantic** -- request/response validation
