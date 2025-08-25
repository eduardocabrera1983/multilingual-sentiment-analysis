#!/usr/bin/env python3
"""
Flask App for Multilingual Sentiment Analysis - EC2 Fixed Version
This version has corrected import paths and CPU compatibility
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import io
import base64

# Fix import paths
project_root = Path(__file__).parent
if project_root.name == 'web_app':
    project_root = project_root.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

print(f"[INFO] Project root: {project_root}")
print(f"[INFO] Python path updated")

# Import your enhanced models - FIXED PATH
try:
    from src.models.enhanced_sentiment_classifier import EnhancedSentimentClassifier
    from src.models.enhanced_aspect_classifier import EnhancedAspectClassifier
    from src.integrated_ml_pipeline import IntegratedMLPipeline  # FIXED: correct path
    MODELS_AVAILABLE = True
    print("[SUCCESS] Successfully imported all enhanced models!")
except ImportError as e:
    print(f"[WARNING] Could not import enhanced models: {e}")
    print("[INFO] The app will run in basic mode")
    MODELS_AVAILABLE = False

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'multilingual-sentiment-key-2025')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DEBUG'] = os.environ.get('FLASK_ENV') != 'production'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
ml_pipeline = None
model_loading_status = {'loaded': False, 'error': None}

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_ml_models():
    """Load ML models with CPU fallback"""
    global ml_pipeline, model_loading_status
    
    try:
        if MODELS_AVAILABLE:
            print("[LOADING] Loading Enhanced ML Pipeline...")
            # Force CPU mode for EC2
            if os.environ.get('FORCE_CPU', 'false').lower() == 'true':
                print("[INFO] Forcing CPU mode for EC2 deployment")
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
            ml_pipeline = IntegratedMLPipeline()
            model_loading_status = {'loaded': True, 'error': None}
            print("[SUCCESS] Enhanced ML Pipeline loaded successfully!")
            
        else:
            model_loading_status = {'loaded': False, 'error': 'Enhanced models not available'}
            print("[WARNING] Running without enhanced models")
            
    except Exception as e:
        error_msg = f"Failed to load ML models: {str(e)}"
        model_loading_status = {'loaded': False, 'error': error_msg}
        logger.error(error_msg)
        print(f"[ERROR] {error_msg}")

# Load models when module imports
load_ml_models()

@app.route('/health')
def health_check():
    """Health check endpoint for EC2"""
    return jsonify({
        'status': 'healthy',
        'service': 'multilingual-sentiment-analysis',
        'models_loaded': model_loading_status['loaded'],
        'timestamp': datetime.now().isoformat(),
        'deployment': 'ec2-cpu'
    }), 200

@app.route('/')
def index():
    """Homepage"""
    stats = {
        'languages_supported': 5,
        'models_loaded': 'Enhanced Pipeline' if model_loading_status['loaded'] else 'Basic Pipeline',
        'features': [
            'Multi-label Aspect Classification',
            'User Experience Prioritization',
            'Business Intelligence Generation'
        ]
    }
    return render_template('index.html', stats=stats, model_status=model_loading_status)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze_text():
    """Single text analysis"""
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        language = request.form.get('language', 'auto')
        
        if not text:
            flash('Please enter text to analyze', 'error')
            return render_template('analyze.html', model_status=model_loading_status)
        
        try:
            start_time = time.time()
            
            if ml_pipeline and model_loading_status['loaded']:
                result = ml_pipeline.analyze_text(text, language)
                analysis_type = 'Enhanced Multi-Label Analysis'
            else:
                result = basic_analysis_fallback(text)
                analysis_type = 'Basic Analysis'
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['analysis_type'] = analysis_type
            
            return render_template('results.html',
                                 text=text,
                                 result=result,
                                 single_analysis=True,
                                 model_status=model_loading_status)
                                 
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            flash(f'Analysis failed: {str(e)}', 'error')
            return render_template('analyze.html', model_status=model_loading_status)
    
    return render_template('analyze.html', model_status=model_loading_status)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for analysis"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', 'auto')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if ml_pipeline and model_loading_status['loaded']:
            result = ml_pipeline.analyze_text(text, language)
            return jsonify(result), 200
        else:
            return jsonify({'error': 'Models not loaded'}), 503
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def basic_analysis_fallback(text):
    """Basic fallback analysis"""
    return {
        'text': text,
        'sentiment': 'neutral',
        'sentiment_confidence': 0.5,
        'primary_aspect': 'general_satisfaction',
        'secondary_aspects': [],
        'classification_type': 'single_aspect',
        'priority_level': 'MEDIUM',
        'business_summary': 'Basic analysis - enhanced models loading...'
    }

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    print("[INFO] Starting Flask App")
    print(f"[INFO] Models Available: {MODELS_AVAILABLE}")
    print(f"[INFO] Model Status: {model_loading_status}")
    app.run(debug=False, host='0.0.0.0', port=5000)