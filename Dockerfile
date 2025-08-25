FROM python:3.9-slim

RUN apt-get update && apt-get install -y gcc g++ curl git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CPU="true"
ENV FLASK_ENV="production"

RUN mkdir -p /app/uploads /app/.cache /app/web_app/uploads

EXPOSE 5000

HEALTHCHECK CMD curl -f http://localhost:5000/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
