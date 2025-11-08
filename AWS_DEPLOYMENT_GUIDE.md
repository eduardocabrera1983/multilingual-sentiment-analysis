# Customer Voice ML - AWS EC2 Deployment Guide

## 🚀 Complete AWS EC2 Deployment Process

### Step 1: Launch EC2 Instance

1. **Go to AWS Console** → EC2 → Launch Instance
2. **Choose AMI**: Ubuntu Server 22.04 LTS
3. **Instance Type**: t3.medium (2 vCPU, 4 GB RAM) - minimum recommended
4. **Storage**: 20 GB gp3 SSD minimum
5. **Security Group**: Create new with these rules:
   ```
   Type            Protocol    Port Range    Source
   SSH             TCP         22            YOUR_IP/32
   HTTP            TCP         80            0.0.0.0/0
   HTTPS           TCP         443           0.0.0.0/0
   Custom TCP      TCP         5000          YOUR_IP/32
   ```
   
   **How to find YOUR_IP:**
   - **Method 1**: Visit https://whatismyipaddress.com/
   - **Method 2**: PowerShell: `(Invoke-WebRequest -uri "http://ifconfig.me/ip").Content`
   - **Method 3**: Browser: https://icanhazip.com/
   - **Example**: If your IP is 31.151.174.197, use `31.151.174.197/32`
6. **Key Pair**: Create or select existing key pair
7. **Launch Instance**

### Step 2: Connect to EC2 Instance

```bash
# Replace with your EC2 PUBLIC IP and key file
ssh -i customervoice-ml-key.pem ubuntu@YOUR_EC2_PUBLIC_IP

# Example:
# ssh -i customervoice-ml-key.pem ubuntu@18.206.107.24
```

**Important**: 
- Use the **PUBLIC IP** (not private IP) from your EC2 dashboard
- The public IP looks like: `18.206.107.24` (example)
- The private IP looks like: `172.31.x.x` (this won't work from outside AWS)

### Step 3: Initial Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Download and run setup script
wget https://raw.githubusercontent.com/eduardocabrera1983/multilingual-sentiment-analysis/main/ec2_setup.sh
chmod +x ec2_setup.sh
./ec2_setup.sh

# IMPORTANT: Log out and back in for Docker permissions
exit
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### Step 4: Deploy Application

```bash
# Run deployment script
~/deploy-customervoice-ml.sh

# Check if application is running
curl http://localhost/health
```

### Step 5: Configure Domain (Hostinger)

1. **Login to Hostinger Control Panel**
2. **Go to Domain → Manage → DNS Records**
3. **Add/Update A Records**:
   ```
   Type    Name    Content             TTL
   A       @       YOUR_EC2_IP         3600
   A       www     YOUR_EC2_IP         3600
   ```
4. **Wait for DNS propagation** (5-30 minutes)

### Step 6: Setup SSL Certificate

```bash
# Download SSL setup script
wget https://raw.githubusercontent.com/eduardocabrera1983/multilingual-sentiment-analysis/main/setup_ssl.sh
chmod +x setup_ssl.sh

# Run SSL setup (replace email with your email)
sudo ./setup_ssl.sh
```

### Step 7: Verify Deployment

1. **Test HTTP**: `curl http://customervoice-ml.com/health`
2. **Test HTTPS**: `curl https://customervoice-ml.com/health`
3. **Browser Test**: Visit https://customervoice-ml.com

## 🔧 Management Commands

### Monitor Application
```bash
# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Monitor resources
htop
```

### Update Application
```bash
cd /opt/customervoice-ml
git pull origin main
./deploy_app.sh
```

### Restart Services
```bash
docker-compose restart
```

### Backup Data
```bash
# Backup uploads and data
tar -czf backup-$(date +%Y%m%d).tar.gz uploads/ data/ cache/
```

## 🛡️ Security Best Practices

### Firewall Configuration
```bash
sudo ufw status
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### Regular Updates
```bash
# Set up automatic updates
sudo apt install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

## 📊 Monitoring & Alerts

### CloudWatch Setup (Optional)
1. Install CloudWatch agent
2. Monitor CPU, Memory, Disk usage
3. Set up alerts for high resource usage

### Application Monitoring
```bash
# Check application health
curl https://customervoice-ml.com/health

# Monitor container resources
docker stats
```

## 🚨 Troubleshooting

### Common Issues

1. **Port 80/443 already in use**:
   ```bash
   sudo netstat -tulpn | grep :80
   sudo systemctl stop apache2  # If Apache is running
   ```

2. **Docker permission denied**:
   ```bash
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

3. **SSL certificate issues**:
   ```bash
   sudo certbot certificates
   sudo certbot renew --dry-run
   ```

4. **Application not starting**:
   ```bash
   docker-compose logs customervoice-ml
   docker-compose restart
   ```

5. **DNS not propagating**:
   - Wait up to 24 hours for global propagation
   - Use online DNS propagation checkers
   - Clear local DNS cache

## 💰 Cost Optimization

### EC2 Instance Recommendations:
- **Development**: t3.small ($15/month)
- **Production**: t3.medium ($30/month)
- **High Traffic**: t3.large ($60/month)

### Reserved Instances:
- Save up to 75% with 1-3 year commitments
- Consider for production environments

## 🎯 Production Checklist

- [ ] EC2 instance launched and configured
- [ ] Security group properly configured
- [ ] Domain DNS pointing to EC2 IP
- [ ] SSL certificate installed and working
- [ ] Application health check passing
- [ ] Monitoring and alerts configured
- [ ] Backup strategy implemented
- [ ] Documentation updated with server details

## 📞 Support

For deployment issues, check:
1. Application logs: `docker-compose logs`
2. System resources: `htop` and `df -h`
3. Network connectivity: `curl` tests
4. DNS resolution: `nslookup customervoice-ml.com`