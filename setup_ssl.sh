#!/bin/bash

# SSL Certificate Setup for Customer Voice ML
# This script configures Let's Encrypt SSL certificates

set -e

echo "🔒 Customer Voice ML - SSL Certificate Setup"
echo "============================================="

DOMAIN="customervoice-ml.com"
EMAIL="contact@customervoice-ml.com"  # Replace with your email

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    log_error "Please run this script with sudo"
    exit 1
fi

# Install Certbot if not already installed
if ! command -v certbot &> /dev/null; then
    log_info "Installing Certbot..."
    apt-get update
    apt-get install -y certbot python3-certbot-nginx
fi

# Stop nginx temporarily
log_info "Stopping nginx container for certificate generation..."
docker-compose stop nginx 2>/dev/null || true

# Generate certificate using standalone mode
log_info "Generating SSL certificate for $DOMAIN..."
certbot certonly --standalone \
    --email $EMAIL \
    --agree-tos \
    --no-eff-email \
    -d $DOMAIN \
    -d www.$DOMAIN

# Create SSL directory
mkdir -p ./ssl

# Copy certificates to application directory
log_info "Copying certificates..."
cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem ./ssl/cert.pem
cp /etc/letsencrypt/live/$DOMAIN/privkey.pem ./ssl/key.pem

# Set proper permissions
chown -R 1000:1000 ./ssl
chmod 644 ./ssl/cert.pem
chmod 600 ./ssl/key.pem

# Start nginx with SSL
log_info "Starting nginx with SSL configuration..."
docker-compose up -d nginx

# Setup certificate renewal
log_info "Setting up automatic certificate renewal..."
(crontab -l 2>/dev/null; echo "0 12 * * * /usr/bin/certbot renew --quiet --deploy-hook '/usr/local/bin/docker-compose -f $(pwd)/docker-compose.yml restart nginx'") | crontab -

# Test certificate
log_info "Testing SSL certificate..."
sleep 10
if curl -f https://$DOMAIN/health >/dev/null 2>&1; then
    log_success "SSL certificate configured successfully!"
    echo ""
    echo "🎉 Your site is now available at:"
    echo "   • https://$DOMAIN"
    echo "   • https://www.$DOMAIN"
    echo ""
    echo "🔄 Certificate will auto-renew every 3 months"
else
    log_warning "SSL test failed, but certificates are installed"
    echo "Please check your DNS configuration and firewall settings"
fi

echo ""
log_info "Certificate information:"
certbot certificates