#!/bin/bash
echo "Deploying application..."
mkdir -p uploads cache
docker-compose down 2>/dev/null
docker-compose build
docker-compose up -d
sleep 30
if curl -f http://localhost/health; then
    echo "[OK] Deployed successfully!"
    echo "Access at: http://$(curl -s ifconfig.me)"
else
    echo "[ERROR] Health check failed. Check: docker-compose logs"
fi
