#!/bin/bash

# Customer Voice ML Deployment Script
# This script deploys the application to production

set -e

echo "🌐 Customer Voice ML - Production Deployment"
echo "============================================="

# Configuration
REPO_URL="https://github.com/eduardocabrera1983/multilingual-sentiment-analysis.git"
APP_DIR="/opt/customervoice-ml"
DOMAIN="customervoice-ml.com"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Create application directory
log_info "Creating application directory..."
mkdir -p uploads cache data logs

# Update from git if in git directory
if [ -d ".git" ]; then
    log_info "Updating application code..."
    git pull origin main
fi

# Stop existing containers
log_info "Stopping existing services..."
docker-compose down 2>/dev/null || true

# Build new image
log_info "Building Customer Voice ML Docker image..."
docker-compose build --no-cache

# Start services
log_info "Starting Customer Voice ML services..."
docker-compose up -d

# Wait for startup
log_info "Waiting for services to initialize..."
sleep 60

# Health check
log_info "Performing health check..."
for i in {1..10}; do
    if curl -f http://localhost/health >/dev/null 2>&1; then
        log_success "Customer Voice ML is running successfully!"
        break
    elif [ $i -eq 10 ]; then
        log_error "Health check failed after 10 attempts"
        docker-compose logs
        exit 1
    else
        log_info "Health check attempt $i/10..."
        sleep 10
    fi
done

# Show status
log_info "Deployment Status:"
docker-compose ps

# Get public IP
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "unknown")

log_success "🎉 Customer Voice ML deployed successfully!"
echo ""
echo "📊 Access your application:"
echo "   • Local: http://localhost"
echo "   • Public IP: http://$PUBLIC_IP"
echo "   • Domain: https://$DOMAIN (after DNS configuration)"
echo ""
echo "🔧 Management commands:"
echo "   • Logs: docker-compose logs -f"
echo "   • Restart: docker-compose restart"
echo "   • Stop: docker-compose down"
echo "   • Update: ./deploy_app.sh"
echo ""
