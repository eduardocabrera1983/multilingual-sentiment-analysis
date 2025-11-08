# Customer Voice ML - Production Docker Image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Production environment variables
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CPU="true"
ENV FLASK_ENV="production"
ENV PYTHONPATH="/app"
ENV DOMAIN="customervoice-ml.com"

# Create necessary directories
RUN mkdir -p /app/uploads /app/.cache /app/web_app/uploads /app/data

# Set proper permissions
RUN chmod +x /app/app.py

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Production command with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "--max-requests", "1000", "--preload", "app:app"]
