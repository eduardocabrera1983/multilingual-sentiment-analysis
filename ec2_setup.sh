#!/bin/bash
echo "🌐 Setting up Customer Voice ML on EC2..."

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install essential packages
sudo apt-get install -y curl wget git unzip software-properties-common

# Install Docker
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "🐙 Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Create swap for better performance
if [ ! -f /swapfile ]; then
    echo "💾 Creating swap file..."
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# Install Certbot for SSL certificates
echo "🔒 Installing Certbot for SSL..."
sudo apt-get install -y certbot python3-certbot-nginx

# Create necessary directories
sudo mkdir -p /opt/customervoice-ml/{ssl,logs,data,uploads,cache}
sudo chown -R $USER:$USER /opt/customervoice-ml

# Configure firewall
echo "🛡️ Configuring firewall..."
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# Install monitoring tools
echo "📊 Installing monitoring tools..."
sudo apt-get install -y htop iotop nethogs

# Create deployment script
cat > ~/deploy-customervoice-ml.sh << 'EOF'
#!/bin/bash
echo "🚀 Deploying Customer Voice ML..."

# Clone or update repository
if [ ! -d "multilingual-sentiment-analysis" ]; then
    git clone https://github.com/eduardocabrera1983/multilingual-sentiment-analysis.git
else
    cd multilingual-sentiment-analysis
    git pull origin main
    cd ..
fi

cd multilingual-sentiment-analysis

# Build and start services
echo "🔨 Building Docker image..."
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 60

# Check health
echo "🏥 Checking application health..."
curl -f http://localhost/health && echo "✅ Application is healthy!" || echo "❌ Application health check failed!"

echo "🎉 Deployment complete!"
echo "🌐 Your site will be available at: https://customervoice-ml.com"
EOF

chmod +x ~/deploy-customervoice-ml.sh

echo ""
echo "✅ EC2 Setup Complete!"
echo "📋 Next Steps:"
echo "  1. Log out and log back in for Docker permissions"
echo "  2. Run: ~/deploy-customervoice-ml.sh"
echo "  3. Configure SSL: sudo certbot --nginx -d customervoice-ml.com -d www.customervoice-ml.com"
echo "  4. Point your domain DNS to this EC2 instance IP"
echo ""
