# Fabric Consumption Forecasting System

**Version:** 1.0.0
**Developer:** Azim Mahmud
**Release Date:** January 2026

A production-ready Streamlit dashboard for intelligent fabric consumption prediction with dual unit support (Meters & Yards).

---

## Features

- **AI-Powered Predictions:** Machine learning models for accurate fabric consumption forecasting
- **Dual Unit Support:** Seamless conversion between meters and yards
- **Batch Processing:** Upload CSV files for bulk predictions
- **ROI Calculator:** Analyze economic impact and environmental benefits
- **Production-Ready:** Comprehensive error handling, logging, and configuration management

---

## Quick Start

### Local Development

```bash
# Clone repository
git clone <repository-url>
cd fabric-forecast-app

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access at http://localhost:8501
```

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Description | Default |
|----------|-------------|---------|
| `FABRIC_APP_ENV` | Environment (development/production) | `production` |
| `FABRIC_APP_LOG_LEVEL` | Logging level | `INFO` |
| `FABRIC_APP_MAX_FILE_SIZE_MB` | Upload file size limit | `10` |
| `FABRIC_APP_MAX_BATCH_ROWS` | Batch row limit | `1000` |
| `FABRIC_APP_MODEL_PATH` | Model directory path | `models` |
| `FABRIC_APP_SESSION_TIMEOUT_MINUTES` | Session timeout | `120` |

---

## Production Deployment

### Using Docker

```bash
# Build image
docker build -t fabric-forecast:1.0.0 .

# Run container
docker run -d \
  --name fabric-forecast \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/models:/app/models:ro \
  fabric-forecast:1.0.0
```

### Using Docker Compose

```bash
docker-compose up -d
```

### Kubernetes (Helm)

```bash
helm install fabric-forecast ./helm-chart
```

---

## Project Structure

```
fabric-forecast-app/
├── app.py                 # Main application
├── config.toml            # Streamlit configuration
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container image definition
├── docker-compose.yml    # Multi-container orchestration
├── .env.example          # Environment template
├── .dockerignore         # Docker exclusions
├── models/               # ML model files
│   ├── xgboost_model.pkl
│   ├── random_forest_model.pkl
│   ├── linear_regression_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── model_metadata.json
└── logs/                 # Application logs
```

---

## API Reference

### Prediction Result Schema

```python
{
    "prediction": float,           # Primary prediction value
    "prediction_alternate": float, # Value in alternate unit
    "unit": str,                   # "meters" or "yards"
    "unit_alternate": str,         # Alternate unit name
    "confidence_lower": float,     # Lower confidence bound
    "confidence_upper": float,     # Upper confidence bound
    "model_name": str,             # Model used
    "timestamp": str               # ISO format timestamp
}
```

---

## Monitoring & Logging

### Log Format

```
2026-01-29 12:34:56 | INFO | app | main:100 | Application started
```

### Health Check

```
GET /_stcore/health
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Models not loading | Ensure models are in `/app/models` directory |
| File upload fails | Check `FABRIC_APP_MAX_FILE_SIZE_MB` setting |
| Session expires | Adjust `FABRIC_APP_SESSION_TIMEOUT_MINUTES` |

---

## License

Proprietary - All Rights Reserved

© 2026 Azim Mahmud. Fabric Consumption Forecasting System.
