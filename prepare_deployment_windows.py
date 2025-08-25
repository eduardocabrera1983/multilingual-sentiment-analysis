#!/usr/bin/env python3
"""
Simple EC2 Deployment Preparation Script - Windows Compatible
Run this locally to prepare everything for EC2 deployment using existing models
"""

import os
import shutil
import subprocess
from pathlib import Path

def create_deployment_package():
    """Create deployment package with all necessary files"""
    
    print("\n" + "="*60)
    print("PREPARING EC2 DEPLOYMENT (Using Existing Models)")
    print("="*60)
    
    # Get project root
    project_root = Path.cwd()
    
    # Step 1: Check if required directories exist
    print("\nChecking project structure...")
    required_dirs = ['src', 'src/models', 'web_app']
    missing_dirs = []
    
    for dir_path in required_dirs:
        if not (project_root / dir_path).exists():
            missing_dirs.append(dir_path)
            print(f"  [X] Missing: {dir_path}")
        else:
            print(f"  [OK] Found: {dir_path}")
    
    if missing_dirs:
        print("\n[ERROR] Missing required directories!")
        print("Please ensure your project has the correct structure.")
        return False
    
    # Step 2: Check for model files - FIXED PATHS
    print("\nChecking model files...")
    
    # Check both possible locations for integrated_ml_pipeline.py
    integrated_pipeline_locations = [
        'src/integrated_ml_pipeline.py',
        'src/pipelines/integrated_ml_pipeline.py',
        'integrated_ml_pipeline.py'
    ]
    
    integrated_found = None
    for location in integrated_pipeline_locations:
        if (project_root / location).exists():
            integrated_found = location
            print(f"  [OK] Found integrated_ml_pipeline.py at: {location}")
            break
    
    if not integrated_found:
        print(f"  [WARNING] integrated_ml_pipeline.py not found")
        print("  Checking if it needs to be moved...")
    
    # Check other model files
    model_files = [
        'src/models/enhanced_sentiment_classifier.py',
        'src/models/enhanced_aspect_classifier.py'
    ]
    
    for file_path in model_files:
        if not (project_root / file_path).exists():
            print(f"  [X] Missing: {file_path}")
        else:
            print(f"  [OK] Found: {file_path}")
    
    # Step 2.5: Move integrated_ml_pipeline.py if needed
    if integrated_found and integrated_found != 'src/integrated_ml_pipeline.py':
        print(f"\nMoving integrated_ml_pipeline.py to correct location...")
        source = project_root / integrated_found
        dest = project_root / 'src' / 'integrated_ml_pipeline.py'
        try:
            shutil.copy2(source, dest)
            print(f"  [OK] Copied to src/integrated_ml_pipeline.py")
        except Exception as e:
            print(f"  [WARNING] Could not copy: {e}")
    
    # Step 3: Fix app.py if it exists
    print("\nFixing app.py import paths...")
    app_paths = [
        project_root / 'app.py',
        project_root / 'web_app' / 'app.py'
    ]
    
    app_fixed = False
    for app_path in app_paths:
        if app_path.exists():
            content = app_path.read_text(encoding='utf-8')
            # Fix the import path
            if 'from src.pipelines.integrated_ml_pipeline' in content:
                content = content.replace(
                    'from src.pipelines.integrated_ml_pipeline import IntegratedMLPipeline',
                    'from src.integrated_ml_pipeline import IntegratedMLPipeline'
                )
                app_path.write_text(content, encoding='utf-8')
                print(f"  [OK] Fixed import in {app_path.name}")
                app_fixed = True
            else:
                print(f"  [OK] {app_path.name} already correct")
            break
    
    if not app_fixed:
        print("  [INFO] Creating new app.py with correct imports")
        # Create a minimal app.py
        app_content = '''#!/usr/bin/env python3
"""Flask App for Sentiment Analysis"""

import os
import sys
from pathlib import Path
from flask import Flask, jsonify

# Fix imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

try:
    from src.models.enhanced_sentiment_classifier import EnhancedSentimentClassifier
    from src.models.enhanced_aspect_classifier import EnhancedAspectClassifier
    from src.integrated_ml_pipeline import IntegratedMLPipeline
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Models not available: {e}")
    MODELS_AVAILABLE = False

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "models": MODELS_AVAILABLE})

@app.route('/')
def index():
    return "<h1>Sentiment Analysis API</h1><p>Models: {}</p>".format(MODELS_AVAILABLE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
'''
        (project_root / 'app.py').write_text(app_content, encoding='utf-8')
        print("  [OK] Created new app.py")
    
    # Step 4: Create deployment scripts WITHOUT EMOJIS
    print("\nCreating deployment scripts...")
    
    # EC2 Setup Script - NO EMOJIS
    setup_script = '''#!/bin/bash
echo "Setting up EC2 instance..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Create swap
if [ ! -f /swapfile ]; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

echo "[OK] Setup complete! Please log out and back in for Docker permissions."
'''
    
    # Deploy Script - NO EMOJIS
    deploy_script = '''#!/bin/bash
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
'''
    
    # Write scripts with UTF-8 encoding
    (project_root / 'ec2_setup.sh').write_text(setup_script, encoding='utf-8')
    (project_root / 'deploy_app.sh').write_text(deploy_script, encoding='utf-8')
    
    # Make executable on Unix systems
    try:
        os.chmod(project_root / 'ec2_setup.sh', 0o755)
        os.chmod(project_root / 'deploy_app.sh', 0o755)
    except:
        pass  # Windows doesn't support chmod
    
    print("  [OK] Created ec2_setup.sh")
    print("  [OK] Created deploy_app.sh")
    
    # Step 5: Create Dockerfile
    print("\nCreating Dockerfile...")
    dockerfile = '''FROM python:3.9-slim

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
'''
    
    (project_root / 'Dockerfile').write_text(dockerfile, encoding='utf-8')
    print("  [OK] Created Dockerfile")
    
    # Step 6: Create docker-compose.yml
    print("\nCreating docker-compose.yml...")
    compose = '''version: '3.8'

services:
  sentiment-app:
    build: .
    ports:
      - "80:5000"
    environment:
      - FLASK_ENV=production
      - FORCE_CPU=true
    volumes:
      - ./uploads:/app/uploads
      - ./cache:/app/.cache
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 6G
'''
    
    (project_root / 'docker-compose.yml').write_text(compose, encoding='utf-8')
    print("  [OK] Created docker-compose.yml")
    
    # Step 7: Create requirements.txt
    print("\nCreating requirements.txt...")
    requirements = '''pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
transformers>=4.30.0
flask>=2.0.0
gunicorn>=20.1.0
werkzeug>=2.0.0
langdetect
python-dotenv
'''
    
    (project_root / 'requirements.txt').write_text(requirements, encoding='utf-8')
    print("  [OK] Created requirements.txt")
    
    # Step 8: Create .dockerignore
    print("\nCreating .dockerignore...")
    dockerignore = '''__pycache__
*.pyc
*.pyo
*.pyd
.Python
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
.pytest_cache/
nosetests.xml
coverage.xml
*.cover
*.log
.git
.gitignore
.mypy_cache
.hypothesis
.ipynb_checkpoints
*.ipynb
data/
*.csv
*.xlsx
.env
.venv
venv/
ENV/
'''
    
    (project_root / '.dockerignore').write_text(dockerignore, encoding='utf-8')
    print("  [OK] Created .dockerignore")
    
    # Step 9: Create deployment package (Windows compatible)
    print("\nCreating deployment package...")
    
    # For Windows, create a ZIP file instead of tar.gz
    if os.name == 'nt':  # Windows
        try:
            import zipfile
            
            zip_path = project_root / 'deployment.zip'
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files except excluded ones
                exclude_dirs = {'.git', '__pycache__', 'data', '.ipynb_checkpoints', 'venv', '.venv'}
                exclude_extensions = {'.pyc', '.pyo', '.ipynb', '.log'}
                
                for root, dirs, files in os.walk(project_root):
                    # Remove excluded directories from traversal
                    dirs[:] = [d for d in dirs if d not in exclude_dirs]
                    
                    for file in files:
                        if not any(file.endswith(ext) for ext in exclude_extensions):
                            file_path = Path(root) / file
                            arcname = file_path.relative_to(project_root)
                            zipf.write(file_path, arcname)
            
            print(f"  [OK] Created deployment.zip")
            package_created = True
            package_name = "deployment.zip"
        except Exception as e:
            print(f"  [WARNING] Could not create zip: {e}")
            package_created = False
            package_name = None
    else:  # Unix/Linux
        try:
            subprocess.run([
                'tar', '-czf', 'deployment.tar.gz',
                '--exclude=*.pyc', '--exclude=__pycache__', 
                '--exclude=.git', '--exclude=data',
                '--exclude=*.ipynb', '--exclude=.env',
                '.'
            ], check=True)
            print("  [OK] Created deployment.tar.gz")
            package_created = True
            package_name = "deployment.tar.gz"
        except:
            print("  [WARNING] Could not create tar.gz")
            package_created = False
            package_name = None
    
    # Print summary
    print("\n" + "="*60)
    print("[SUCCESS] DEPLOYMENT PREPARATION COMPLETE!")
    print("="*60)
    
    print("\nFiles created:")
    print("  - Dockerfile")
    print("  - docker-compose.yml")
    print("  - requirements.txt")
    print("  - ec2_setup.sh")
    print("  - deploy_app.sh")
    print("  - .dockerignore")
    print("  - app.py (if missing)")
    if package_created:
        print(f"  - {package_name}")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    
    print("\n1. LAUNCH EC2 INSTANCE:")
    print("   - Type: t3.large (recommended)")
    print("   - OS: Ubuntu 20.04")
    print("   - Security: Open ports 22 (SSH) and 80 (HTTP)")
    
    print("\n2. TRANSFER FILES TO EC2:")
    if package_created:
        if os.name == 'nt':
            print(f"   scp -i your-key.pem {package_name} ubuntu@ec2-ip:~/")
        else:
            print(f"   scp -i your-key.pem {package_name} ubuntu@ec2-ip:~/")
    else:
        print("   scp -r -i your-key.pem . ubuntu@ec2-ip:~/app/")
    
    print("\n3. CONNECT TO EC2:")
    print("   ssh -i your-key.pem ubuntu@ec2-ip")
    
    print("\n4. ON EC2, RUN:")
    if package_created:
        if package_name == "deployment.zip":
            print("   unzip deployment.zip")
        else:
            print("   tar -xzf deployment.tar.gz")
    print("   chmod +x ec2_setup.sh deploy_app.sh")
    print("   ./ec2_setup.sh")
    print("   # Log out and back in for Docker permissions")
    print("   exit")
    print("   ssh -i your-key.pem ubuntu@ec2-ip")
    print("   ./deploy_app.sh")
    
    print("\n5. ACCESS YOUR APP:")
    print("   http://your-ec2-ip")
    
    print("\n" + "="*60)
    print("Good luck with your deployment!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    try:
        success = create_deployment_package()
        if not success:
            print("\n[ERROR] Preparation failed. Please fix the issues above.")
            exit(1)
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()
        exit(1)