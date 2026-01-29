# ============================================================================
# FABRIC CONSUMPTION FORECASTING SYSTEM
# Production Dockerfile
# ============================================================================
# Version: 1.0.0
# Developer: Azim Mahmud
# ============================================================================

# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_LOGGER_LEVEL=info \
    FABRIC_APP_ENV=production \
    FABRIC_APP_LOG_LEVEL=INFO

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY config.toml ./

# Create models directory (empty - models to be mounted or downloaded)
RUN mkdir -p /app/models

# Create logs directory
RUN mkdir -p /app/logs

# Expose Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
