#!/bin/bash
# SSL Setup Script for Customer Voice ML Platform

echo "======================================================================="
echo "SSL SETUP FOR CUSTOMER VOICE ML"
echo "======================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "Please run this script with sudo"
    exit 1
fi

print_status "Step 1: Updating system packages..."
apt update -y

print_status "Step 2: Installing Certbot and Nginx..."
apt install certbot python3-certbot-nginx nginx -y

print_status "Step 3: Stopping Docker containers temporarily..."
cd /home/ubuntu/multilingual-sentiment-analysis
docker-compose down

print_status "Step 4: Creating directory for Let's Encrypt challenge..."
mkdir -p /var/www/certbot

print_status "Step 5: Starting Nginx for certificate validation..."
systemctl stop nginx
systemctl start nginx
systemctl enable nginx

print_status "Step 6: Obtaining SSL certificate from Let's Encrypt..."
print_warning "You will be prompted to enter your email and agree to terms..."

# Get SSL certificate
certbot certonly --webroot \
    -w /var/www/certbot \
    -d customervoice-ml.com \
    -d www.customervoice-ml.com \
    --email your-email@example.com \
    --agree-tos \
    --non-interactive \
    --no-eff-email

if [ $? -eq 0 ]; then
    print_status "SSL certificate obtained successfully!"
else
    print_error "Failed to obtain SSL certificate. Please check your domain DNS settings."
    exit 1
fi

print_status "Step 7: Stopping system Nginx (Docker will handle it)..."
systemctl stop nginx
systemctl disable nginx

print_status "Step 8: Starting Docker containers with SSL..."
docker-compose up -d

print_status "Step 9: Setting up automatic renewal..."
# Add renewal command to crontab
(crontab -l 2>/dev/null; echo "0 2 * * 0 /usr/bin/certbot renew --quiet && docker-compose -f /home/ubuntu/multilingual-sentiment-analysis/docker-compose.yml restart nginx") | crontab -

print_status "======================================================================="
print_status "SSL SETUP COMPLETE!"
print_status "======================================================================="
print_status "Your site should now be accessible at:"
print_status "https://customervoice-ml.com"
print_status "https://www.customervoice-ml.com"
print_status ""
print_status "SSL certificate will automatically renew every Sunday at 2 AM"
print_status "======================================================================="