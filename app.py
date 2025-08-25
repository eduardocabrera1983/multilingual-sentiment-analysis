#!/usr/bin/env python3
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
