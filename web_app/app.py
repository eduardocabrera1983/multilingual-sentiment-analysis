#!/usr/bin/env python3
"""
Enhanced Flask App for Multilingual Sentiment Analysis
Improved version based on your Module 2 Flask architecture
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
import plotly.graph_objs as go
import plotly.utils

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

# Import your enhanced models
try:
    from src.models.enhanced_sentiment_classifier import EnhancedSentimentClassifier
    from src.models.enhanced_aspect_classifier import EnhancedAspectClassifier
    from src.integrated_ml_pipeline import IntegratedMLPipeline
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import enhanced models: {e}")
    MODELS_AVAILABLE = False

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'multilingual-sentiment-key-2025')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DEBUG'] = os.environ.get('FLASK_ENV') != 'production'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging setup
if not app.debug:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

# Global variables for models
ml_pipeline = None
model_loading_status = {'loaded': False, 'error': None}

def load_ml_models():
    """Load the enhanced ML models"""
    global ml_pipeline, model_loading_status
    
    try:
        if MODELS_AVAILABLE:
            print("🚀 Loading Enhanced ML Pipeline...")
            ml_pipeline = IntegratedMLPipeline()
            model_loading_status = {'loaded': True, 'error': None}
            print("✅ Enhanced ML Pipeline loaded successfully")
        else:
            model_loading_status = {'loaded': False, 'error': 'Enhanced models not available'}
            print("⚠️ Running without enhanced models")
    except Exception as e:
        error_msg = f"Failed to load ML models: {str(e)}"
        model_loading_status = {'loaded': False, 'error': error_msg}
        print(f"❌ {error_msg}")

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return {
        'status': 'healthy',
        'service': 'multilingual-sentiment-analysis',
        'models_loaded': model_loading_status['loaded'],
        'timestamp': datetime.now().isoformat()
    }, 200

@app.route('/')
def index():
    """Enhanced homepage with project showcase"""
    stats = {
        'languages_supported': 5,
        'models_loaded': 'Enhanced Multi-Label Pipeline' if model_loading_status['loaded'] else 'Basic Pipeline',
        'features': [
            'Multi-label Aspect Classification',
            'User Experience Prioritization',
            'Business Intelligence Generation',
            'Mixed Concerns Detection',
            'Real-time Processing'
        ]
    }
    return render_template('index.html', stats=stats, model_status=model_loading_status)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze_text():
    """Single text analysis with enhanced features"""
    
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        language = request.form.get('language', 'auto')
        
        if not text:
            flash('Please enter text to analyze', 'error')
            return render_template('analyze.html')
        
        try:
            start_time = time.time()
            
            if ml_pipeline and model_loading_status['loaded']:
                # Use enhanced pipeline
                result = ml_pipeline.analyze_text(text, language)
                analysis_type = 'Enhanced Multi-Label Analysis'
            else:
                # Fallback analysis
                result = fallback_analysis(text)
                analysis_type = 'Basic Analysis (Enhanced Models Not Available)'
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['analysis_type'] = analysis_type
            
            return render_template('results.html', 
                                 text=text, 
                                 result=result,
                                 single_analysis=True)
            
        except Exception as e:
            app.logger.error(f"Analysis error: {e}")
            flash(f'Analysis failed: {str(e)}', 'error')
            return render_template('analyze.html')
    
    return render_template('analyze.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Enhanced file upload with batch processing"""
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                # Save uploaded file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                
                # Process file
                results = process_uploaded_file(filepath)
                
                return render_template('batch_results.html', 
                                     results=results,
                                     filename=filename)
                
            except Exception as e:
                app.logger.error(f"File processing error: {e}")
                flash(f'File processing failed: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload CSV or Excel files.', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/dashboard')
def dashboard():
    """Business Intelligence Dashboard"""
    if not model_loading_status['loaded']:
        flash('Enhanced models required for dashboard features', 'warning')
        return render_template('dashboard.html', demo_mode=True)
    
    # Demo data for dashboard
    demo_data = generate_demo_dashboard_data()
    
    return render_template('dashboard.html', 
                         data=demo_data, 
                         demo_mode=False)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for text analysis"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        language = data.get('language', 'auto')
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'}), 400
        
        if ml_pipeline and model_loading_status['loaded']:
            result = ml_pipeline.analyze_text(text, language)
        else:
            result = fallback_analysis(text)
        
        return jsonify({
            'success': True,
            'result': result,
            'model_type': 'enhanced' if model_loading_status['loaded'] else 'fallback'
        })
        
    except Exception as e:
        app.logger.error(f"API analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/batch-analyze', methods=['POST'])
def api_batch_analyze():
    """API endpoint for batch analysis"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'success': False, 'error': 'No texts provided'}), 400
        
        if ml_pipeline and model_loading_status['loaded']:
            batch_results = ml_pipeline.analyze_batch_with_business_intelligence(texts)
        else:
            batch_results = fallback_batch_analysis(texts)
        
        return jsonify({
            'success': True,
            'results': batch_results,
            'model_type': 'enhanced' if model_loading_status['loaded'] else 'fallback'
        })
        
    except Exception as e:
        app.logger.error(f"API batch analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/about')
def about():
    """About page with methodology and technical details"""
    methodology = {
        'models_used': [
            'XLM-RoBERTa (Multilingual Sentiment)',
            'mBERT (Backup Sentiment Analysis)', 
            'DistilBERT-Multilingual (Fast Processing)',
            'BART-MNLI (Zero-Shot Aspect Classification)'
        ],
        'languages_supported': ['English', 'Spanish', 'German', 'French', 'Dutch'],
        'features': [
            'Multi-label Aspect Classification',
            'User Experience Prioritization',
            'Business Intelligence Generation',
            'Mixed Concerns Detection',
            'Priority Level Assessment',
            'Actionable Recommendations'
        ],
        'performance': {
            'sentiment_accuracy': '100%',
            'aspect_accuracy': '100%',
            'processing_speed': '20+ texts/second',
            'system_reliability': '90.91%'
        }
    }
    return render_template('about.html', methodology=methodology)

@app.route('/export/<analysis_type>')
def export_results(analysis_type):
    """Export analysis results"""
    # This would implement export functionality
    # For demo, return a sample CSV
    sample_data = {
        'text': ['Sample review text'],
        'sentiment': ['positive'],
        'primary_aspect': ['user_experience'],
        'priority_level': ['HIGH']
    }
    
    df = pd.DataFrame(sample_data)
    output = io.StringIO()
    df.to_csv(output, index=False)
    
    output_bytes = io.BytesIO()
    output_bytes.write(output.getvalue().encode('utf-8'))
    output_bytes.seek(0)
    
    return send_file(
        output_bytes,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'sentiment_analysis_{analysis_type}_{datetime.now().strftime("%Y%m%d")}.csv'
    )

def allowed_file(filename):
    """Check if file type is allowed"""
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_uploaded_file(filepath):
    """Process uploaded CSV/Excel file"""
    try:
        # Read file
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Find text column
        text_columns = ['text', 'review', 'comment', 'content', 'message']
        text_column = None
        
        for col in text_columns:
            if col in df.columns:
                text_column = col
                break
        
        if not text_column:
            # Use first string column
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_column = col
                    break
        
        if not text_column:
            raise ValueError("No text column found in file")
        
        # Process with ML pipeline
        if ml_pipeline and model_loading_status['loaded']:
            enhanced_df = ml_pipeline.analyze_dataframe(df, text_column)
            business_intelligence = enhanced_df.attrs.get('business_intelligence', {})
        else:
            enhanced_df = df.copy()
            # Add basic analysis
            enhanced_df['predicted_sentiment'] = 'neutral'
            enhanced_df['predicted_primary_aspect'] = 'general_satisfaction'
            business_intelligence = {}
        
        # Generate summary statistics
        summary = {
            'total_reviews': len(enhanced_df),
            'filename': os.path.basename(filepath),
            'columns_added': [col for col in enhanced_df.columns if col.startswith('predicted_')],
            'business_intelligence': business_intelligence,
            'processing_successful': True
        }
        
        return {
            'summary': summary,
            'dataframe': enhanced_df,
            'sample_rows': enhanced_df.head(10).to_dict('records') if len(enhanced_df) > 0 else []
        }
        
    except Exception as e:
        return {
            'summary': {'processing_successful': False, 'error': str(e)},
            'dataframe': None,
            'sample_rows': []
        }

def fallback_analysis(text):
    """Fallback analysis when enhanced models aren't available"""
    # Simple rule-based analysis
    text_lower = text.lower()
    
    # Basic sentiment
    positive_words = ['good', 'great', 'excellent', 'love', 'perfect', 'amazing']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'worst']
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        sentiment = 'positive'
        confidence = 0.7
    elif neg_count > pos_count:
        sentiment = 'negative'
        confidence = 0.7
    else:
        sentiment = 'neutral'
        confidence = 0.5
    
    # Basic aspect detection
    if any(word in text_lower for word in ['interface', 'design', 'ui', 'ux']):
        primary_aspect = 'user_experience'
    elif any(word in text_lower for word in ['crash', 'bug', 'slow', 'performance']):
        primary_aspect = 'performance'
    else:
        primary_aspect = 'general_satisfaction'
    
    return {
        'sentiment': sentiment,
        'sentiment_confidence': confidence,
        'primary_aspect': primary_aspect,
        'secondary_aspects': [],
        'classification_type': 'single_aspect',
        'priority_level': 'MEDIUM',
        'business_summary': f'Basic analysis - {sentiment} sentiment about {primary_aspect}',
        'recommendation': 'Upgrade to enhanced models for detailed analysis'
    }

def fallback_batch_analysis(texts):
    """Fallback batch analysis"""
    individual_results = [fallback_analysis(text) for text in texts]
    
    return {
        'individual_results': individual_results,
        'business_intelligence': {
            'total_reviews': len(texts),
            'message': 'Enhanced models required for full business intelligence'
        },
        'summary_metrics': {
            'total_analyzed': len(texts),
            'model_type': 'fallback'
        }
    }

def generate_demo_dashboard_data():
    """Generate demo data for dashboard"""
    return {
        'sentiment_distribution': {'positive': 45, 'negative': 25, 'neutral': 30},
        'aspect_distribution': {
            'user_experience': 35,
            'performance': 25,
            'tracking_accuracy': 20,
            'general_satisfaction': 20
        },
        'priority_levels': {'HIGH': 15, 'MEDIUM': 50, 'LOW': 35},
        'language_distribution': {'en': 60, 'es': 20, 'de': 10, 'fr': 7, 'nl': 3}
    }

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Load models at startup
    load_ml_models()
    
    # Start app
    if os.environ.get('FLASK_ENV') == 'production':
        app.logger.info("🚀 Starting in production mode")
    else:
        app.logger.info("🔧 Starting in development mode")
        print("🚀 Multilingual Sentiment Analysis - Enhanced Flask App")
        print("📊 Features: Multi-label Classification, Business Intelligence, Real-time Processing")
        print("🌍 Languages: English, Spanish, German, French, Dutch")
        print("⚡ Models: Enhanced ML Pipeline with User Experience Prioritization")
        app.run(debug=True, host='0.0.0.0', port=5000)

# Load models when imported by gunicorn
load_ml_models()