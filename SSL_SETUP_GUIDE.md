# SSL Setup Guide for Customer Voice ML

## Prerequisites
- Domain pointing to your EC2 IP: customervoice-ml.com → 176.34.205.198
- SSH access to your EC2 server

## Step-by-step SSL Setup

### 1. Connect to your EC2 server
```bash
ssh -i "B:\IronHack\Keys\customervoice-ml-key.pem" ubuntu@176.34.205.198
```

### 2. Update system and install Certbot
```bash
sudo apt update
sudo apt install certbot python3-certbot-nginx -y
```

### 3. Stop Docker containers temporarily
```bash
cd multilingual-sentiment-analysis
sudo docker-compose down
```

### 4. Create webroot directory for Let's Encrypt
```bash
sudo mkdir -p /var/www/certbot
```

### 5. Start temporary nginx for certificate validation
```bash
sudo systemctl start nginx
```

### 6. Obtain SSL certificate (replace with your email)
```bash
sudo certbot certonly --webroot \
    -w /var/www/certbot \
    -d customervoice-ml.com \
    -d www.customervoice-ml.com \
    --email your-email@example.com \
    --agree-tos \
    --non-interactive
```

### 7. Stop system nginx (Docker will handle nginx)
```bash
sudo systemctl stop nginx
sudo systemctl disable nginx
```

### 8. Pull latest code with SSL configuration
```bash
git pull origin main
```

### 9. Start Docker containers with SSL
```bash
sudo docker-compose up -d
```

### 10. Verify SSL is working
```bash
curl -I https://customervoice-ml.com
```

### 11. Setup automatic renewal (optional)
```bash
echo "0 2 * * 0 /usr/bin/certbot renew --quiet && docker-compose -f /home/ubuntu/multilingual-sentiment-analysis/docker-compose.yml restart nginx" | sudo crontab -
```

## Expected Results
- ✅ https://customervoice-ml.com works with green lock
- ✅ https://www.customervoice-ml.com works with green lock  
- ✅ http://customervoice-ml.com redirects to https://
- ✅ SSL certificate auto-renewal configured

## Troubleshooting
- If certificate fails: Check DNS propagation and security groups
- If containers don't start: Check docker logs with `sudo docker-compose logs`
- If SSL doesn't work: Verify certificate paths in nginx.conf