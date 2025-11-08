#!/usr/bin/env python3
"""
Production-Ready Flask App for Multilingual Sentiment Analysis
UPDATED for Two-Model Ensemble Integration + Auto-FedEx Analysis
Supports both CPU and GPU with automatic detection and manual override
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[SUCCESS] Environment variables loaded from .env file")
except ImportError:
    print("[INFO] python-dotenv not installed. Install with: pip install python-dotenv")
    print("[INFO] Environment variables will be loaded from system environment only")
except Exception as e:
    print(f"[WARNING] Could not load .env file: {e}")
    print("[INFO] Using system environment variables only")

# =====================================================
# CUSTOM JSON ENCODER - SOLUTION FOR SERIALIZATION
# =====================================================
from flask.json.provider import DefaultJSONProvider

class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle objects that aren't natively JSON serializable
    Handles: LineData objects, numpy types, datetime objects, pandas objects, custom classes
    """
    def default(self, obj):
        try:
            # Handle numpy types
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            
            # Handle datetime objects
            elif isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            elif isinstance(obj, timedelta):
                return obj.total_seconds()
            
            # Handle pandas objects
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            
            # Handle custom objects with __dict__ (like LineData)
            elif hasattr(obj, '__dict__'):
                # Convert custom objects to dictionary
                result = {}
                for key, value in obj.__dict__.items():
                    # Skip private attributes and methods
                    if not key.startswith('_'):
                        try:
                            # Recursively serialize the value
                            result[key] = json.loads(json.dumps(value, cls=CustomJSONEncoder))
                        except (TypeError, ValueError):
                            # If we can't serialize the value, convert to string
                            result[key] = str(value)
                return result
            
            # Handle objects with to_dict method
            elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                return obj.to_dict()
            
            # Handle objects with to_json method
            elif hasattr(obj, 'to_json') and callable(getattr(obj, 'to_json')):
                return json.loads(obj.to_json())
            
            # Handle sets
            elif isinstance(obj, set):
                return list(obj)
            
            # Handle complex numbers
            elif isinstance(obj, complex):
                return {'real': obj.real, 'imag': obj.imag}
            
            # Handle pathlib Path objects
            elif isinstance(obj, Path):
                return str(obj)
            
            # For any other object, try to convert to string as last resort
            else:
                return str(obj)
                
        except Exception as e:
            # If all else fails, return a string representation
            return f"<Non-serializable: {type(obj).__name__}>"

class CustomJSONProvider(DefaultJSONProvider):
    """
    Custom JSON provider for Flask that uses our custom encoder
    This ensures both API responses AND template rendering use our custom serialization
    """
    def dumps(self, obj, **kwargs):
        return json.dumps(obj, cls=CustomJSONEncoder, **kwargs)
    
    def loads(self, s):
        return json.loads(s)

# Configure device preference from environment
FORCE_CPU = os.environ.get('FORCE_CPU', 'false').lower() == 'true'
FORCE_GPU = os.environ.get('FORCE_GPU', 'false').lower() == 'true'

# Fix import paths
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

print(f"[INFO] Project root: {project_root}")
print(f"[INFO] Templates: {project_root / 'web_app' / 'templates'}")
print(f"[INFO] Static files: {project_root / 'web_app' / 'static'}")

# Device configuration
if FORCE_CPU:
    print("[CONFIG] Forcing CPU mode (FORCE_CPU=true)")
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU for PyTorch
    device_mode = 'cpu'
elif FORCE_GPU:
    print("[CONFIG] Forcing GPU mode (FORCE_GPU=true)")
    device_mode = 'gpu'
else:
    print("[CONFIG] Auto-detecting best device...")
    device_mode = 'auto'

# Try importing PyTorch to check GPU availability
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available() and not FORCE_CPU
    if GPU_AVAILABLE:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[SUCCESS] GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("[INFO] Running on CPU")
except ImportError:
    GPU_AVAILABLE = False
    print("[INFO] PyTorch not installed - running on CPU")

# Import enhanced models with two-model ensemble support
try:
    from src.integrated_ml_pipeline import IntegratedMLPipeline
    from src.models.enhanced_sentiment_classifier import EnhancedSentimentClassifier
    from src.models.enhanced_aspect_classifier import EnhancedAspectClassifier
    MODELS_AVAILABLE = True
    print("[SUCCESS] Enhanced models with two-model ensemble imported successfully!")
except ImportError as e:
    print(f"[WARNING] Could not import enhanced models: {e}")
    MODELS_AVAILABLE = False

# NEW: Import FedEx analyzer for auto-analysis
try:
    from src.scrapers.fedex_scraper import FedExReviewAnalyzer
    FEDEX_SCRAPER_AVAILABLE = True
    print("[SUCCESS] FedEx scraper imported successfully!")
except ImportError as e:
    FEDEX_SCRAPER_AVAILABLE = False
    print(f"[WARNING] FedEx scraper not available: {e}")

# Initialize Flask with correct paths
app = Flask(__name__, 
           template_folder='web_app/templates',
           static_folder='web_app/static')

# =====================================================
# APPLY CUSTOM JSON PROVIDER TO FLASK APP
# =====================================================
app.json = CustomJSONProvider(app)
print("[SUCCESS] Custom JSON provider applied to Flask app (handles both API + templates)")

# Configuration with secure environment variable handling
def get_secret_key():
    """Generate or retrieve a secure secret key"""
    secret_key = os.environ.get('SECRET_KEY')
    if not secret_key:
        # Generate a random secret key for development
        import secrets
        secret_key = secrets.token_hex(32)
        print(f"[WARNING] No SECRET_KEY found in environment. Generated temporary key.")
        print(f"[SECURITY] For production, set SECRET_KEY environment variable!")
    return secret_key

app.config['SECRET_KEY'] = get_secret_key()
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_UPLOAD_SIZE_MB', '16')) * 1024 * 1024
app.config['UPLOAD_FOLDER'] = str(project_root / os.environ.get('UPLOAD_DIR', 'uploads'))
app.config['CACHE_FOLDER'] = str(project_root / os.environ.get('CACHE_DIR', 'cache'))
app.config['DATA_FOLDER'] = str(project_root / os.environ.get('DATA_DIR', 'data'))

# Production settings from environment
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

# CORS and security settings
app.config['CORS_ORIGINS'] = os.environ.get('CORS_ORIGINS', '*')
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SESSION_COOKIE_SECURE', 'false').lower() == 'true'
app.config['SESSION_COOKIE_HTTPONLY'] = os.environ.get('SESSION_COOKIE_HTTPONLY', 'true').lower() == 'true'

# Create necessary directories
for folder in ['uploads', 'cache', 'data']:
    os.makedirs(project_root / folder, exist_ok=True)

# Global variables (EXISTING)
ml_pipeline = None
model_status = {'loaded': False, 'error': None, 'device': 'unknown', 'ensemble_info': {}}

# NEW: Global variables for auto-analysis
fedex_analyzer = None
analysis_status = {
    'running': False,
    'last_run': None,
    'last_file': None,
    'total_reviews': 0,
    'error': None
}

# NEW: Data refresh tracking
data_refresh_status = {
    'last_check': datetime.now(),
    'latest_file': None,
    'needs_refresh': False
}

# Logging configuration
log_level = logging.DEBUG if app.config['DEBUG'] else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================
# HELPER FUNCTION FOR SAFE JSON RESPONSES
# =====================================================
def safe_jsonify(*args, **kwargs):
    """
    Wrapper around jsonify that uses our custom encoder
    This provides extra safety for any edge cases
    """
    try:
        return jsonify(*args, **kwargs)
    except TypeError as e:
        logger.error(f"JSON serialization error: {e}")
        # Return error response instead of crashing
        error_response = {
            'error': 'Serialization failed',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(error_response), 500

def clean_data_for_template(data):
    """
    Clean data dictionary to ensure all values are JSON serializable for templates
    This is an extra safety layer for template rendering
    """
    if not isinstance(data, dict):
        return data
    
    cleaned = {}
    for key, value in data.items():
        if isinstance(value, timedelta):
            cleaned[key] = value.total_seconds()
        elif isinstance(value, (datetime, pd.Timestamp)):
            cleaned[key] = value.isoformat()
        elif isinstance(value, dict):
            cleaned[key] = clean_data_for_template(value)
        elif isinstance(value, list):
            cleaned[key] = [clean_data_for_template(item) if isinstance(item, dict) else 
                           (item.total_seconds() if isinstance(item, timedelta) else 
                           (item.isoformat() if isinstance(item, (datetime, pd.Timestamp)) else item)) 
                           for item in value]
        elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool)):
            # Convert custom objects to dict
            cleaned[key] = clean_data_for_template(value.__dict__)
        else:
            cleaned[key] = value
    
    return cleaned

# EXISTING FUNCTION (unchanged)
def load_ml_models():
    """Load ML models with two-model ensemble support and device flexibility"""
    global ml_pipeline, model_status
    
    try:
        if not MODELS_AVAILABLE:
            raise ImportError("Enhanced models not available")
        
        print("\n[LOADING] Initializing ML Pipeline with Two-Model Ensemble...")
        
        # Determine device to use
        if FORCE_CPU:
            device = 'cpu'
            device_desc = "CPU (forced)"
        elif FORCE_GPU and GPU_AVAILABLE:
            device = 'cuda'
            device_desc = "GPU (forced)"
        elif GPU_AVAILABLE and not FORCE_CPU:
            device = 'cuda'
            device_desc = f"GPU ({torch.cuda.get_device_name(0)})"
        else:
            device = 'cpu'
            device_desc = "CPU"
        
        print(f"[INFO] Using device: {device_desc}")
        
        # Initialize pipeline with two-model ensemble support
        start_time = time.time()
        ml_pipeline = IntegratedMLPipeline(device=device, verbose=True)
        load_time = time.time() - start_time
        
        # Get ensemble information if available
        ensemble_info = {}
        try:
            pipeline_info = ml_pipeline.get_pipeline_info()
            ensemble_info = {
                'version': pipeline_info.get('version', 'unknown'),
                'pipeline_type': pipeline_info.get('pipeline_type', 'unknown'),
                'models_loaded': pipeline_info.get('models_loaded', {}),
                'features': pipeline_info.get('features', [])
            }
            
            # Get sentiment classifier details if available
            if hasattr(ml_pipeline, 'sentiment_classifier') and ml_pipeline.sentiment_classifier:
                sentiment_info = ml_pipeline.sentiment_classifier.get_model_info()
                ensemble_info['sentiment_details'] = {
                    'version': sentiment_info.get('version', 'unknown'),
                    'ensemble_enabled': sentiment_info.get('ensemble_enabled', False),
                    'loaded_models': sentiment_info.get('loaded_models', 0),
                    'models': sentiment_info.get('models', {})
                }
        except Exception as e:
            print(f"[WARNING] Could not get ensemble info: {e}")
        
        model_status = {
            'loaded': True, 
            'error': None,
            'device': device_desc,
            'load_time': load_time,
            'ensemble_info': ensemble_info
        }
        
        print(f"[SUCCESS] ML Pipeline with Two-Model Ensemble loaded in {load_time:.2f} seconds")
        print(f"[INFO] Running on: {device_desc}")
        
        # Print ensemble details
        if ensemble_info.get('sentiment_details', {}).get('ensemble_enabled'):
            ensemble_models = ensemble_info['sentiment_details'].get('loaded_models', 0)
            print(f"[INFO] Two-Model Ensemble: {ensemble_models} models loaded")
        
    except Exception as e:
        error_msg = str(e)
        model_status = {
            'loaded': False, 
            'error': error_msg,
            'device': 'none',
            'ensemble_info': {}
        }
        logger.error(f"Failed to load models: {error_msg}")
        print(f"[ERROR] Model loading failed: {error_msg}")

# NEW: FedEx analyzer initialization
def initialize_fedex_analyzer():
    """Initialize FedEx analyzer"""
    global fedex_analyzer
    
    if not FEDEX_SCRAPER_AVAILABLE:
        logger.warning("FedEx scraper not available")
        return False
        
    try:
        fedex_analyzer = FedExReviewAnalyzer(
            data_dir=str(project_root / 'data'),
            use_enhanced_models=MODELS_AVAILABLE and model_status['loaded'],
            device='auto'
        )
        logger.info("FedEx analyzer initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize FedEx analyzer: {e}")
        return False

# NEW: Check for existing FedEx data (7-day freshness)
def check_existing_fedex_data():
    """Check if FedEx analysis data already exists"""
    data_dir = Path(app.config['DATA_FOLDER'])
    cache_dir = Path(app.config['CACHE_FOLDER'])
    
    # Look for FedEx-specific files
    fedex_patterns = [
        'fedex_reviews_enhanced_ensemble_*.csv',
        'fedex_reviews_*.csv'
    ]
    
    for directory in [data_dir, cache_dir]:
        for pattern in fedex_patterns:
            files = list(directory.glob(pattern))
            if files:
                latest = max(files, key=lambda p: p.stat().st_mtime)
                file_age = datetime.now() - datetime.fromtimestamp(latest.stat().st_mtime)
                
                # Consider data fresh if less than 7 days old
                if file_age < timedelta(days=7):
                    logger.info(f"Found fresh FedEx data: {latest.name} ({file_age.days} days, {file_age.seconds//3600} hours old)")
                    return latest, len(pd.read_csv(latest)) if latest.exists() else 0
                else:
                    logger.info(f"Found stale FedEx data: {latest.name} ({file_age.days} days old)")
    
    return None, 0

# NEW: Run FedEx analysis in background
def run_fedex_analysis_background(target_reviews=1000):
    """Run FedEx analysis in background thread"""
    global analysis_status
    
    def analysis_thread():
        try:
            analysis_status['running'] = True
            analysis_status['error'] = None
            logger.info(f"Starting background FedEx analysis for {target_reviews} reviews")
            
            if not fedex_analyzer:
                if not initialize_fedex_analyzer():
                    raise Exception("Failed to initialize FedEx analyzer")
            
            # Run the analysis
            df = fedex_analyzer.analyze_fedex_reviews(
                count=target_reviews,
                real_only=True
            )
            
            if df is not None and len(df) > 0:
                analysis_status['last_run'] = datetime.now()
                analysis_status['total_reviews'] = len(df)
                
                # Find the generated file
                data_dir = Path(app.config['DATA_FOLDER'])
                fedex_files = list(data_dir.glob('fedex_reviews_enhanced_ensemble_*.csv'))
                if fedex_files:
                    latest = max(fedex_files, key=lambda p: p.stat().st_mtime)
                    analysis_status['last_file'] = str(latest)
                    data_refresh_status['needs_refresh'] = True
                    logger.info(f"FedEx analysis completed: {len(df)} reviews saved to {latest.name}")
                else:
                    logger.warning("FedEx analysis completed but file not found")
            else:
                raise Exception("FedEx analysis returned no data")
                
        except Exception as e:
            analysis_status['error'] = str(e)
            logger.error(f"Background FedEx analysis failed: {e}")
        finally:
            analysis_status['running'] = False
    
    # Start the background thread
    thread = threading.Thread(target=analysis_thread)
    thread.daemon = True
    thread.start()
    
    return thread

# NEW: Get latest analysis data with latest time prioritization
def get_latest_analysis_data():
    """Get the most recent analysis data - FIXED VERSION"""
    data_sources = []
    
    # Find all CSV files in both directories
    data_dir = Path(app.config['DATA_FOLDER'])
    cache_dir = Path(app.config['CACHE_FOLDER'])
    
    print(f"[DEBUG] Searching for data files...")
    print(f"[DEBUG] Data dir: {data_dir}")
    print(f"[DEBUG] Cache dir: {cache_dir}")
    
    # Collect all CSV files with their types and modification times
    for directory in [data_dir, cache_dir]:
        if directory.exists():
            print(f"[DEBUG] Checking directory: {directory}")
            
            # FedEx ensemble files (highest priority)
            for f in directory.glob('fedex_reviews_enhanced_ensemble_*.csv'):
                print(f"[DEBUG] Found FedEx ensemble file: {f.name}")
                data_sources.append((f, 'FedEx Real Data', f.stat().st_mtime))
            
            # Regular FedEx files (exclude ensemble to avoid duplicates)
            for f in directory.glob('fedex_reviews_*.csv'):
                if 'enhanced_ensemble' not in f.name:
                    print(f"[DEBUG] Found FedEx file: {f.name}")
                    data_sources.append((f, 'FedEx Data', f.stat().st_mtime))
            
            # Ensemble results
            for f in directory.glob('results_ensemble_*.csv'):
                print(f"[DEBUG] Found ensemble results: {f.name}")
                data_sources.append((f, 'Ensemble Results', f.stat().st_mtime))
            
            # Regular results  
            for f in directory.glob('results_*.csv'):
                print(f"[DEBUG] Found regular results: {f.name}")
                data_sources.append((f, 'Analysis Results', f.stat().st_mtime))
        else:
            print(f"[DEBUG] Directory does not exist: {directory}")
    
    if not data_sources:
        print("[DEBUG] No CSV files found in expected locations")
        return None, True
    
    # Sort by modification time - NEWEST FIRST (highest priority)
    data_sources.sort(key=lambda x: x[2], reverse=True)
    
    print(f"[DEBUG] Found {len(data_sources)} potential data files:")
    for i, (path, source, mtime) in enumerate(data_sources[:3]):  # Show top 3
        file_time = datetime.fromtimestamp(mtime)
        print(f"[DEBUG]   {i+1}. {source}: {path.name} ({file_time})")
    
    # Try to load files in order (newest first)
    for data_path, source_name, mtime in data_sources:
        if data_path.exists():
            try:
                print(f"[DEBUG] Attempting to load: {data_path.name}")
                
                # Enhanced CSV reading with better error handling
                df = pd.read_csv(data_path, encoding='utf-8')
                print(f"[DEBUG] Successfully read {len(df)} rows, {len(df.columns)} columns")
                
                # RELAXED validation - be less strict about column requirements
                if len(df.columns) < 3:  # Very basic sanity check
                    print(f"[WARNING] Skipping {data_path.name} - too few columns ({len(df.columns)})")
                    continue
                
                # Check if this looks like a properly formatted file
                expected_cols = ['sentiment', 'primary_aspect', 'classification_type', 'priority_level']
                has_predicted_cols = any(col.startswith('predicted_') for col in df.columns)
                has_original_cols = any(col in df.columns for col in expected_cols)
                
                # IMPROVED: Also check for partial matches
                has_some_expected = any(
                    any(expected in col.lower() for expected in ['sentiment', 'aspect', 'classification', 'priority'])
                    for col in df.columns
                )
                
                if not (has_original_cols or has_predicted_cols or has_some_expected):
                    print(f"[WARNING] Skipping {data_path.name} - missing expected columns")
                    print(f"[DEBUG] Available columns: {list(df.columns)[:10]}...")
                    continue
                
                if len(df) == 0:
                    print(f"[WARNING] Skipping {data_path.name} - empty dataframe")
                    continue
                
                print(f"[SUCCESS] Loading data from: {data_path.name}")
                print(f"[DEBUG] Columns found: {list(df.columns)}")
                
                # Generate dashboard data
                data = generate_dashboard_data(df)
                data['source'] = source_name
                data['file_path'] = str(data_path)
                data['file_age'] = datetime.now() - datetime.fromtimestamp(data_path.stat().st_mtime)
                data['is_fedex_data'] = 'fedex' in data_path.name.lower()
                
                # Log which file was actually loaded with timestamp
                file_time = datetime.fromtimestamp(mtime)
                logger.info(f"Dashboard loaded: {source_name} from {data_path.name} ({len(df)} reviews) - Modified: {file_time}")
                print(f"[SUCCESS] Dashboard will show {data['total_reviews']} reviews from {source_name}")
                
                return data, False  # Not demo mode
                
            except Exception as e:
                print(f"[ERROR] Could not load {data_path}: {e}")
                logger.warning(f"Could not load {data_path}: {e}")
                continue
    
    # No data found
    print("[WARNING] No valid data files found - using demo mode")
    logger.info("No valid data files found - using demo mode")
    return None, True

# NEW: Monitor data changes (daily checks)
def monitor_data_changes():
    """Monitor for new data files and set refresh flag"""
    global data_refresh_status
    
    def monitor_thread():
        while True:
            try:
                # Check for new files once per day
                time.sleep(86400)  # 24 hours = 86400 seconds
                
                current_latest = None
                latest_mtime = 0
                
                # Check all data directories for new files
                for directory in [Path(app.config['DATA_FOLDER']), Path(app.config['CACHE_FOLDER'])]:
                    for pattern in ['*.csv']:
                        for file_path in directory.glob(pattern):
                            mtime = file_path.stat().st_mtime
                            if mtime > latest_mtime:
                                latest_mtime = mtime
                                current_latest = file_path
                
                # Check if we have a new latest file
                if (current_latest != data_refresh_status['latest_file'] and 
                    datetime.fromtimestamp(latest_mtime) > data_refresh_status['last_check']):
                    
                    data_refresh_status['latest_file'] = current_latest
                    data_refresh_status['needs_refresh'] = True
                    data_refresh_status['last_check'] = datetime.now()
                    logger.info(f"Daily check: New data detected - {current_latest.name if current_latest else 'None'}")
                else:
                    logger.info("Daily check: No new data detected")
                
            except Exception as e:
                logger.error(f"Error in daily data monitoring: {e}")
    
    # Start monitoring thread
    thread = threading.Thread(target=monitor_thread)
    thread.daemon = True
    thread.start()
    logger.info("Daily data monitoring started")

# Initialize models on startup (EXISTING)
print("\n" + "="*70)
print("INITIALIZING PRODUCTION ML PIPELINE")
print("Two-Model Ensemble Integration + Auto-FedEx Analysis")
print("="*70)
load_ml_models()

# NEW: Initialize FedEx data checking and monitoring
existing_file, review_count = check_existing_fedex_data()

if existing_file:
    analysis_status['last_file'] = str(existing_file)
    analysis_status['total_reviews'] = review_count
    analysis_status['last_run'] = datetime.fromtimestamp(existing_file.stat().st_mtime)
    print(f"[INFO] Found existing FedEx data: {review_count} reviews")
else:
    print("[INFO] No fresh FedEx data found - will analyze on first dashboard access")

# NEW: Start data monitoring
monitor_data_changes()

print("="*70 + "\n")

# --- ROUTES (USE SAFE_JSONIFY WHERE NEEDED) ---

# EXISTING ROUTE (enhanced with safe_jsonify for JSON responses)
@app.route('/health')
def health_check():
    """Health check endpoint for monitoring with ensemble information"""
    health_data = {
        'status': 'healthy' if model_status['loaded'] else 'degraded',
        'service': 'multilingual-sentiment-analysis-with-fedex',  # Updated
        'version': '2.1_auto_analysis',  # Updated
        'models_loaded': model_status['loaded'],
        'device': model_status.get('device', 'unknown'),
        'gpu_available': GPU_AVAILABLE,
        'force_cpu': FORCE_CPU,
        'ensemble_enabled': model_status.get('ensemble_info', {}).get('sentiment_details', {}).get('ensemble_enabled', False),
        'ensemble_models_loaded': model_status.get('ensemble_info', {}).get('sentiment_details', {}).get('loaded_models', 0),
        'fedex_analyzer_available': FEDEX_SCRAPER_AVAILABLE,  # NEW
        'analysis_status': {  # NEW
            'running': analysis_status['running'],
            'last_run': analysis_status['last_run'].isoformat() if analysis_status['last_run'] else None,
            'total_reviews': analysis_status['total_reviews'],
            'has_error': analysis_status['error'] is not None
        },
        'timestamp': datetime.now().isoformat()
    }
    
    if GPU_AVAILABLE and 'torch' in sys.modules:
        health_data['gpu_info'] = {
            'name': torch.cuda.get_device_name(0),
            'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'memory_allocated': torch.cuda.memory_allocated() / 1024**3 if model_status['loaded'] else 0
        }
    
    status_code = 200 if model_status['loaded'] else 503
    return safe_jsonify(health_data), status_code

# EXISTING ROUTE (unchanged)
@app.route('/')
def index():
    """Homepage with system status and ensemble information"""
    ensemble_info = model_status.get('ensemble_info', {})
    sentiment_details = ensemble_info.get('sentiment_details', {})
    
    stats = {
        'languages_supported': 5,
        'models_loaded': f"Two-Model Ensemble ({model_status.get('device', 'Unknown')})" if model_status['loaded'] else 'Not Loaded',
        'features': ensemble_info.get('features', [
            'Multi-label Aspect Classification',
            'Two-Model Sentiment Ensemble',
            'Automatic FedEx Review Analysis',  # NEW
            'Real-time Dashboard Updates',      # NEW
            'User Experience Prioritization',
            'Business Intelligence Generation',
            f"{'GPU' if GPU_AVAILABLE and not FORCE_CPU else 'CPU'} Processing"
        ]),
        'device_info': model_status.get('device', 'Unknown'),
        'ensemble_enabled': sentiment_details.get('ensemble_enabled', False),
        'ensemble_models': sentiment_details.get('loaded_models', 0),
        'pipeline_version': ensemble_info.get('version', '2.0'),
        'fedex_analysis': {  # NEW
            'available': FEDEX_SCRAPER_AVAILABLE,
            'running': analysis_status['running'],
            'total_reviews': analysis_status['total_reviews'],
            'last_run': analysis_status['last_run'],
            'has_data': analysis_status['last_file'] is not None
        }
    }
    return render_template('index.html', stats=stats, model_status=model_status)

# EXISTING ROUTE (unchanged)
@app.route('/analyze', methods=['GET', 'POST'])
def analyze_text():
    """Single text analysis endpoint with two-model ensemble support"""
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        language = request.form.get('language', 'auto')
        
        if not text:
            flash('Please enter text to analyze', 'error')
            return render_template('analyze.html', model_status=model_status)
        
        try:
            start_time = time.time()
            
            if ml_pipeline and model_status['loaded']:
                result = ml_pipeline.analyze_text(text, language)
                analysis_type = f'Two-Model Ensemble Analysis ({model_status["device"]})'
                
                # Add ensemble-specific metadata
                result['ensemble_metadata'] = {
                    'sentiment_method': result.get('sentiment_method', 'unknown'),
                    'sentiment_models_used': result.get('sentiment_models_used', 0),
                    'sentiment_device': result.get('sentiment_device', 'unknown'),
                    'sentiment_from_cache': result.get('sentiment_from_cache', False),
                    'pipeline_version': result.get('pipeline_version', '2.0')
                }
            else:
                result = basic_analysis_fallback(text)
                analysis_type = 'Basic Analysis (Fallback)'
            
            # Add metadata
            result['processing_time'] = time.time() - start_time
            result['analysis_type'] = analysis_type
            result['device_used'] = model_status.get('device', 'unknown')
            
            return render_template('results.html',
                                 text=text,
                                 result=result,
                                 single_analysis=True,
                                 model_status=model_status)
                                 
        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            flash(f'Analysis failed: {str(e)}', 'error')
            return render_template('analyze.html', model_status=model_status)
    
    return render_template('analyze.html', model_status=model_status)

# EXISTING ROUTE (enhanced with dashboard refresh)
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Batch file processing endpoint with two-model ensemble optimization"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                # Save uploaded file
                file.save(filepath)
                logger.info(f"File uploaded: {filename}")
                
                # Read the file
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath, encoding='utf-8', error_bad_lines=False)
                else:  # Excel
                    df = pd.read_excel(filepath)
                
                # Find text column
                text_col = find_text_column(df)
                if not text_col:
                    flash('No text column found in file', 'error')
                    return redirect(request.url)
                
                # Process with two-model ensemble pipeline
                if ml_pipeline and model_status['loaded']:
                    logger.info(f"Processing {len(df)} rows from column '{text_col}' with two-model ensemble")
                    
                    # Add progress tracking for large files
                    flash(f'Processing {len(df)} texts with two-model ensemble. This may take a moment...', 'info')
                    
                    # Process the dataframe with ensemble support
                    start_time = time.time()
                    results_df = ml_pipeline.analyze_dataframe(df, text_col)
                    processing_time = time.time() - start_time
                    
                    # Save results
                    result_filename = f'results_ensemble_{timestamp}.csv'
                    result_path = project_root / 'cache' / result_filename
                    results_df.to_csv(result_path, index=False)
                    
                    # NEW: Trigger dashboard refresh
                    data_refresh_status['needs_refresh'] = True
                    data_refresh_status['latest_file'] = result_path
                    
                    logger.info(f"Results saved to {result_filename}")
                    
                    # Calculate summary statistics with ensemble metrics
                    summary_stats = calculate_summary_stats(results_df)
                    ensemble_metrics = calculate_ensemble_metrics(results_df)
                    
                    # Prepare results for display
                    results = {
                        'summary': {
                            'filename': file.filename,
                            'total_reviews': len(df),
                            'processing_successful': True,
                            'processing_time': processing_time,
                            'texts_per_second': len(df) / processing_time if processing_time > 0 else 0,
                            'columns_added': [col for col in results_df.columns if col.startswith('predicted_')],
                            'business_intelligence': results_df.attrs.get('business_intelligence', {}),
                            'ensemble_performance': results_df.attrs.get('ensemble_performance', {}),
                            'dashboard_updated': True,  # NEW
                            **summary_stats,
                            **ensemble_metrics
                        },
                        'sample_rows': results_df.head(50).to_dict('records'),
                        'download_file': result_filename
                    }
                    
                    flash(f'Successfully processed {len(df)} texts in {processing_time:.1f} seconds with two-model ensemble. Dashboard will refresh with new data.', 'success')  # Enhanced message
                    
                    return render_template('batch_results.html', 
                                         results=results,
                                         filename=file.filename,
                                         model_status=model_status)
                else:
                    flash('Models not loaded - cannot process file', 'error')
                    return redirect(request.url)
                    
            except Exception as e:
                logger.error(f"File processing error: {e}", exc_info=True)
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(request.url)
            finally:
                # Clean up uploaded file to save space
                if os.path.exists(filepath):
                    os.remove(filepath)
        else:
            flash('Invalid file type. Please upload CSV or Excel file.', 'error')
            return redirect(request.url)
    
    return render_template('upload.html', model_status=model_status)

# ENHANCED ROUTE (with automatic FedEx analysis)
@app.route('/dashboard')
def dashboard():
    """Enhanced dashboard with automatic FedEx analysis and refresh"""
    global data_refresh_status
    
    # Check if we need to start FedEx analysis
    data, demo_mode = get_latest_analysis_data()
    
    if demo_mode and FEDEX_SCRAPER_AVAILABLE and not analysis_status['running']:
        # No real data and not currently analyzing - start background analysis
        logger.info("No data found - starting automatic FedEx analysis")
        run_fedex_analysis_background(target_reviews=1000)
        
        # Show loading state
        return render_template('dashboard_loading.html', 
                             analysis_status=clean_data_for_template(analysis_status),
                             model_status=clean_data_for_template(model_status))
    
    # Reset refresh flag since we're serving fresh data
    data_refresh_status['needs_refresh'] = False
    
    # Provide demo data if still no real data
    if demo_mode:
        data = {
            'total_reviews': 0,
            'mixed_concerns_pct': '0',
            'ux_priority_pct': '0',
            'high_priority_pct': '0',
            'sentiment_distribution': {'positive': 50, 'negative': 30, 'neutral': 20},
            'source': 'Demo Data - Analysis Starting',
            'ensemble_metrics': {
                'ensemble_usage_pct': '0',
                'cache_hit_rate_pct': '0',
                'avg_processing_time_ms': '0',
                'models_used_avg': '0'
            },
            'is_fedex_data': False
        }
    
    # Clean data for template to ensure JSON serialization works
    clean_data = clean_data_for_template(data) if data else None
    clean_model_status = clean_data_for_template(model_status)
    clean_analysis_status = clean_data_for_template(analysis_status)
    
    return render_template('dashboard.html', 
                         demo_mode=demo_mode,
                         data=clean_data,
                         model_status=clean_model_status,
                         analysis_status=clean_analysis_status)  # NEW

# EXISTING ROUTE (unchanged)
@app.route('/about')
def about():
    """About page with methodology including two-model ensemble details"""
    ensemble_info = model_status.get('ensemble_info', {})
    sentiment_details = ensemble_info.get('sentiment_details', {})
    
    methodology = {
        'ensemble_models': [
            'XLM-RoBERTa (60.0% weight)',
            'Twitter-RoBERTa (40.0% weight)',
            'Advanced Rule-based Fallback'
        ],
        'models_used': [
            'XLM-RoBERTa (Multilingual)',
            'Twitter-RoBERTa (Social Media Optimized)', 
            'BART (Zero-shot Classification)',
            'Enhanced Multi-label Aspect Classifier'
        ],
        'languages_supported': ['English', 'Spanish', 'German', 'French', 'Dutch'],
        'performance': {
            'sentiment_accuracy': '95%+',
            'aspect_accuracy': '90%+',
            'processing_speed': f"{'50+' if GPU_AVAILABLE else '15-20'} texts/second",
            'system_reliability': '95%+',
            'device': model_status.get('device', 'Unknown'),
            'ensemble_enabled': sentiment_details.get('ensemble_enabled', False),
            'cache_optimization': True
        },
        'ensemble_features': [
            'Weighted Model Voting',
            'Dynamic Fallback System', 
            'Cache Optimization',
            'GPU Acceleration',
            'Confidence Calibration',
            'Performance Monitoring'
        ]
    }
    return render_template('about.html', 
                         methodology=methodology,
                         model_status=model_status)

# Excel Template Download Route
@app.route('/download/excel-template')
def download_excel_template():
    """Download Excel template with sample data for batch upload"""
    try:
        # Create sample data that matches our expected format
        sample_data = {
            'review': [
                "Interface is confusing but tracking works perfectly and updates in real-time",
                "Perfect tracking always updated shows correct delivery status", 
                "App crashes frequently terrible user experience overall needs major improvements",
                "Great design beautiful interface easy to navigate and very user-friendly",
                "Mixed concerns about performance but tracking is accurate and reliable",
                "Outstanding service and app performance exceeds all expectations"
            ],
            'rating': [3, 5, 1, 4, 3, 5],
            'date': [
                "2025-08-01", "2025-08-02", "2025-08-03", 
                "2025-08-04", "2025-08-05", "2025-08-06"
            ],
            'user_id': [
                "user_001", "user_002", "user_003", 
                "user_004", "user_005", "user_006"
            ],
            'category': [
                "mobile_app", "mobile_app", "mobile_app", 
                "mobile_app", "web_app", "mobile_app"
            ],
            'platform': [
                "ios", "android", "ios", 
                "android", "desktop", "ios"
            ]
        }
        
        # Create DataFrame
        df = pd.DataFrame(sample_data)
        
        # Create Excel file in memory
        from io import BytesIO
        output = BytesIO()
        
        # Use pandas ExcelWriter with openpyxl engine for rich formatting
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Write main data sheet
            df.to_excel(writer, sheet_name='Sample_Data', index=False)
            
            # Create a metadata sheet with instructions
            metadata = pd.DataFrame({
                'Column': ['review', 'rating', 'date', 'user_id', 'category', 'platform'],
                'Description': [
                    'Text content to analyze (required)',
                    'Numeric rating 1-5 (optional)',
                    'Date in YYYY-MM-DD format (optional)',
                    'Unique user identifier (optional)',
                    'Category or app type (optional)',
                    'Platform: ios, android, web, desktop (optional)'
                ],
                'Type': ['Text', 'Number', 'Date', 'Text', 'Text', 'Text'],
                'Required': ['Yes', 'No', 'No', 'No', 'No', 'No']
            })
            metadata.to_excel(writer, sheet_name='Instructions', index=False)
            
            # Format the worksheets
            workbook = writer.book
            
            # Format Sample_Data sheet
            ws1 = writer.sheets['Sample_Data']
            ws1.column_dimensions['A'].width = 60  # Review column wider
            for col in ['B', 'C', 'D', 'E', 'F']:
                ws1.column_dimensions[col].width = 15
            
            # Format Instructions sheet
            ws2 = writer.sheets['Instructions']
            for col in ['A', 'B', 'C', 'D']:
                ws2.column_dimensions[col].width = 20
        
        output.seek(0)
        
        return send_file(
            output,
            download_name='enhanced_sample_reviews_template.xlsx',
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        print(f"[ERROR] Excel template generation failed: {e}")
        # Fallback to CSV if Excel generation fails
        csv_data = "review,rating,date,user_id,category,platform\n"
        csv_data += '"Interface is confusing but tracking works perfectly",3,2025-08-01,user_001,mobile_app,ios\n'
        csv_data += '"Perfect tracking always updated shows correct delivery status",5,2025-08-02,user_002,mobile_app,android\n'
        
        from io import BytesIO
        output = BytesIO(csv_data.encode('utf-8'))
        return send_file(
            output,
            download_name='enhanced_sample_reviews_fallback.csv',
            as_attachment=True,
            mimetype='text/csv'
        )

# NEW: API endpoints for dashboard status and data (using safe_jsonify)
@app.route('/api/dashboard/status')
def dashboard_status():
    """API endpoint for dashboard status checks"""
    return safe_jsonify({
        'analysis_running': analysis_status['running'],
        'needs_refresh': data_refresh_status['needs_refresh'],
        'last_run': analysis_status['last_run'].isoformat() if analysis_status['last_run'] else None,
        'total_reviews': analysis_status['total_reviews'],
        'error': analysis_status['error'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/dashboard/data')
def dashboard_data_api():
    """API endpoint to get fresh dashboard data"""
    global data_refresh_status
    
    data, demo_mode = get_latest_analysis_data()
    data_refresh_status['needs_refresh'] = False
    
    return safe_jsonify({
        'demo_mode': demo_mode,
        'data': data,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/fedex/analyze', methods=['POST'])
def trigger_fedex_analysis():
    """API endpoint to manually trigger FedEx analysis"""
    if analysis_status['running']:
        return safe_jsonify({
            'error': 'Analysis already running',
            'status': 'running'
        }), 409
    
    # Get parameters
    target_reviews = request.json.get('target_reviews', 1000) if request.is_json else 1000
    
    # Start analysis
    thread = run_fedex_analysis_background(target_reviews)
    
    return safe_jsonify({
        'message': f'FedEx analysis started for {target_reviews} reviews',
        'status': 'started',
        'timestamp': datetime.now().isoformat()
    })

# EXISTING ROUTE (enhanced with safe_jsonify)
@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """REST API endpoint for text analysis with ensemble metadata"""
    try:
        data = request.get_json()
        
        if not data:
            return safe_jsonify({'error': 'No JSON data provided'}), 400
            
        text = data.get('text', '')
        language = data.get('language', 'auto')
        
        if not text:
            return safe_jsonify({'error': 'No text provided'}), 400
        
        if ml_pipeline and model_status['loaded']:
            result = ml_pipeline.analyze_text(text, language)
            
            # Add API-specific metadata
            result['api_metadata'] = {
                'device_used': model_status.get('device', 'unknown'),
                'ensemble_enabled': model_status.get('ensemble_info', {}).get('sentiment_details', {}).get('ensemble_enabled', False),
                'version': '2.0_two_model_ensemble'
            }
            
            return safe_jsonify(result), 200
        else:
            fallback = basic_analysis_fallback(text)
            return safe_jsonify({
                'warning': 'Models not loaded, using fallback',
                'result': fallback
            }), 503
            
    except Exception as e:
        logger.error(f"API error: {e}", exc_info=True)
        return safe_jsonify({'error': str(e)}), 500

# EXISTING ROUTE (enhanced with safe_jsonify)
@app.route('/api/batch', methods=['POST'])
def api_batch_analyze():
    """REST API endpoint for batch analysis with ensemble optimization"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return safe_jsonify({'error': 'No texts provided'}), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return safe_jsonify({'error': 'texts must be a list'}), 400
        
        if ml_pipeline and model_status['loaded']:
            # Use optimized batch processing
            start_time = time.time()
            results = ml_pipeline.analyze_batch(texts)
            processing_time = time.time() - start_time
            
            # Calculate batch ensemble metrics
            ensemble_metrics = {
                'total_processed': len(results),
                'processing_time_seconds': processing_time,
                'throughput_texts_per_second': len(results) / processing_time if processing_time > 0 else 0,
                'ensemble_usage_count': sum(1 for r in results if r.get('sentiment_method') == 'two_model_ensemble'),
                'cache_hits': sum(1 for r in results if r.get('sentiment_from_cache', False)),
                'average_confidence': np.mean([r.get('sentiment_confidence', 0) for r in results])
            }
            
            return safe_jsonify({
                'results': results,
                'count': len(results),
                'device_used': model_status.get('device', 'unknown'),
                'ensemble_metrics': ensemble_metrics
            }), 200
        else:
            return safe_jsonify({'error': 'Models not loaded'}), 503
            
    except Exception as e:
        logger.error(f"Batch API error: {e}", exc_info=True)
        return safe_jsonify({'error': str(e)}), 500

# EXISTING ROUTE (unchanged)
@app.route('/export/results/<analysis_type>')
def export_results(analysis_type):
    """Export analysis results as CSV"""
    try:
        cache_dir = Path(app.config['CACHE_FOLDER'])
        # Prioritize ensemble results
        ensemble_files = list(cache_dir.glob('results_ensemble_*.csv'))
        regular_files = list(cache_dir.glob('results_*.csv'))
        
        files = ensemble_files + regular_files
        
        if files:
            latest = max(files, key=lambda p: p.stat().st_mtime)
            download_name = f'ml_analysis_ensemble_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            
            return send_file(
                str(latest), 
                as_attachment=True, 
                download_name=download_name,
                mimetype='text/csv'
            )
        else:
            flash('No results available for export', 'warning')
            return redirect(url_for('dashboard'))
            
    except Exception as e:
        logger.error(f"Export error: {e}", exc_info=True)
        flash(f'Export failed: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

# EXISTING ROUTE (unchanged)
@app.route('/download/<filename>')
def download_file(filename):
    """Download specific result file"""
    try:
        # Sanitize filename
        filename = secure_filename(filename)
        filepath = Path(app.config['CACHE_FOLDER']) / filename
        
        if filepath.exists():
            return send_file(
                str(filepath), 
                as_attachment=True,
                mimetype='text/csv'
            )
        else:
            flash('File not found', 'error')
            return redirect(url_for('dashboard'))
            
    except Exception as e:
        logger.error(f"Download error: {e}", exc_info=True)
        flash(f'Download failed: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

# --- HELPER FUNCTIONS (ALL EXISTING + SOME ENHANCEMENTS) ---

# EXISTING FUNCTION (unchanged)
def basic_analysis_fallback(text):
    """Fallback analysis when models aren't loaded"""
    return {
        'text': text,
        'sentiment': 'neutral',
        'sentiment_confidence': 0.5,
        'sentiment_method': 'basic_fallback',
        'sentiment_models_used': 0,
        'primary_aspect': 'general_satisfaction',
        'secondary_aspects': [],
        'classification_type': 'single_aspect',
        'priority_level': 'MEDIUM',
        'severity_level': 'MODERATE',
        'business_summary': 'Basic fallback analysis',
        'recommendation': 'Load two-model ensemble for detailed analysis',
        'requires_immediate_action': False,
        'user_experience_priority': False,
        'mixed_concerns': False,
        'ensemble_metadata': {
            'sentiment_method': 'basic_fallback',
            'sentiment_models_used': 0,
            'sentiment_device': 'cpu',
            'sentiment_from_cache': False,
            'pipeline_version': 'fallback'
        }
    }

# EXISTING FUNCTION (unchanged)
def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'tsv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# EXISTING FUNCTION (unchanged)
def find_text_column(df):
    """Intelligently find the text column in a dataframe"""
    # Common text column names
    text_columns = ['text', 'review', 'comment', 'content', 'message', 'feedback', 'description']
    
    for col in text_columns:
        if col in df.columns.str.lower():
            # Find the actual column name (case-insensitive match)
            actual_col = [c for c in df.columns if c.lower() == col][0]
            return actual_col
    
    # Look for columns with string data and sufficient length
    for col in df.columns:
        if df[col].dtype == object:
            # Check if this column has text-like content
            sample = df[col].dropna().head(10)
            if len(sample) > 0:
                avg_length = sample.astype(str).str.len().mean()
                if avg_length > 20:  # Likely to be text content
                    return col
    
    return None

# EXISTING FUNCTION (unchanged)
def calculate_summary_stats(df):
    """Calculate summary statistics from results dataframe"""
    stats = {}
    
    if 'predicted_sentiment' in df.columns:
        sentiment_counts = df['predicted_sentiment'].value_counts(normalize=True) * 100
        stats['sentiment_distribution'] = sentiment_counts.to_dict()
    
    if 'predicted_classification_type' in df.columns:
        mixed = (df['predicted_classification_type'] == 'mixed_concerns').mean() * 100
        stats['mixed_concerns_percentage'] = round(mixed, 1)
    
    if 'predicted_primary_aspect' in df.columns:
        aspects = df['predicted_primary_aspect'].value_counts().head(5)
        stats['top_aspects'] = aspects.to_dict()
    
    if 'predicted_priority_level' in df.columns:
        priority = df['predicted_priority_level'].value_counts(normalize=True) * 100
        stats['priority_distribution'] = priority.to_dict()
    
    return stats

# EXISTING FUNCTION (unchanged)
def calculate_ensemble_metrics(df):
    """Calculate two-model ensemble specific metrics"""
    metrics = {}
    
    if 'predicted_sentiment_method' in df.columns:
        method_counts = df['predicted_sentiment_method'].value_counts(normalize=True) * 100
        metrics['ensemble_method_distribution'] = method_counts.to_dict()
        
        ensemble_usage = (df['predicted_sentiment_method'] == 'two_model_ensemble').mean() * 100
        metrics['ensemble_usage_percentage'] = round(ensemble_usage, 1)
    
    if 'predicted_sentiment_models_used' in df.columns:
        avg_models = df['predicted_sentiment_models_used'].mean()
        metrics['average_models_used'] = round(avg_models, 2)
    
    if 'predicted_sentiment_from_cache' in df.columns:
        cache_rate = df['predicted_sentiment_from_cache'].mean() * 100
        metrics['cache_hit_rate_percentage'] = round(cache_rate, 1)
    
    return metrics

# ENHANCED FUNCTION (works with both existing and FedEx data)
def generate_dashboard_data(df):
    """COMPLETELY FIXED: Generate dashboard data with correct percentages and validation"""
    
    print(f"[DEBUG] Generating dashboard data from {len(df)} rows")
    
    # Initialize with defaults
    data = {
        'total_reviews': len(df),
        'mixed_concerns_pct': '0.0',
        'ux_priority_pct': '0.0',
        'high_priority_pct': '0.0',
        'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
        'aspect_distribution': {},
        'priority_levels': {},
        'language_distribution': {},
        'classification_distribution': {},
        'ensemble_metrics': {
            'ensemble_usage_pct': '0.0',
            'cache_hit_rate_pct': '0.0',
            'avg_processing_time_ms': '0.0',
            'models_used_avg': '0.0'
        }
    }
    
    print(f"[DEBUG] Available columns: {list(df.columns)}")
    
    try:
        total_reviews = len(df)
        
        # =====================================================================
        # SENTIMENT DISTRIBUTION (FIXED PERCENTAGES)
        # =====================================================================
        sentiment_cols = ['predicted_sentiment', 'sentiment']
        sentiment_col = None
        for col in sentiment_cols:
            if col in df.columns:
                sentiment_col = col
                break
        
        if sentiment_col:
            sentiment_counts = df[sentiment_col].value_counts()
            
            # Calculate percentages correctly (as integers, not raw counts)
            pos_count = sentiment_counts.get('positive', 0)
            neg_count = sentiment_counts.get('negative', 0)
            neu_count = sentiment_counts.get('neutral', 0)
            
            pos_pct = int((pos_count / total_reviews) * 100) if total_reviews > 0 else 0
            neg_pct = int((neg_count / total_reviews) * 100) if total_reviews > 0 else 0
            neu_pct = int((neu_count / total_reviews) * 100) if total_reviews > 0 else 0
            
            # Ensure percentages add up to 100% (handle rounding)
            total_pct = pos_pct + neg_pct + neu_pct
            if total_pct != 100 and total_reviews > 0:
                # Adjust the largest category to make it sum to 100
                if pos_count >= neg_count and pos_count >= neu_count:
                    pos_pct += (100 - total_pct)
                elif neg_count >= neu_count:
                    neg_pct += (100 - total_pct)
                else:
                    neu_pct += (100 - total_pct)
            
            data['sentiment_distribution'] = {
                'positive': pos_pct,
                'negative': neg_pct,
                'neutral': neu_pct
            }
            
            print(f"[DEBUG] Sentiment - Positive: {pos_count} ({pos_pct}%), Negative: {neg_count} ({neg_pct}%), Neutral: {neu_count} ({neu_pct}%)")
        
        # =====================================================================
        # ASPECT DISTRIBUTION (FIXED)
        # =====================================================================
        aspect_cols = ['predicted_primary_aspect', 'primary_aspect', 'aspect']
        aspect_col = None
        for col in aspect_cols:
            if col in df.columns:
                aspect_col = col
                break
        
        if aspect_col:
            # Get top 6 aspects for the chart
            aspect_counts = df[aspect_col].value_counts().head(6)
            data['aspect_distribution'] = aspect_counts.to_dict()
            
            # Calculate UX priority percentage (correctly as percentage, not count)
            ux_count = (df[aspect_col] == 'user_experience').sum()
            ux_pct = (ux_count / total_reviews) * 100 if total_reviews > 0 else 0
            data['ux_priority_pct'] = f"{ux_pct:.1f}"
            
            print(f"[DEBUG] UX priority: {ux_count}/{total_reviews} = {ux_pct:.1f}%")
            print(f"[DEBUG] Top aspects: {dict(aspect_counts)}")
        
        # =====================================================================
        # PRIORITY LEVELS (FIXED)
        # =====================================================================
        priority_cols = ['predicted_priority_level', 'priority_level', 'priority']
        priority_col = None
        for col in priority_cols:
            if col in df.columns:
                priority_col = col
                break
        
        if priority_col:
            priority_counts = df[priority_col].value_counts()
            data['priority_levels'] = priority_counts.to_dict()
            
            # Calculate high priority percentage (correctly as percentage)
            high_count = (df[priority_col] == 'HIGH').sum()
            high_pct = (high_count / total_reviews) * 100 if total_reviews > 0 else 0
            data['high_priority_pct'] = f"{high_pct:.1f}"
            
            print(f"[DEBUG] High priority: {high_count}/{total_reviews} = {high_pct:.1f}%")
            print(f"[DEBUG] Priority distribution: {dict(priority_counts)}")
        
        # =====================================================================
        # CLASSIFICATION TYPE (FIXED)
        # =====================================================================
        classification_cols = ['predicted_classification_type', 'classification_type', 'type']
        classification_col = None
        for col in classification_cols:
            if col in df.columns:
                classification_col = col
                break
        
        if classification_col:
            # Map classification types to user-friendly names
            classification_counts = df[classification_col].value_counts()
            
            mixed_count = classification_counts.get('mixed_concerns', 0)
            dual_count = classification_counts.get('dual_concerns', 0)  
            single_count = classification_counts.get('single_concern', 0)
            
            mixed_pct = (mixed_count / total_reviews) * 100 if total_reviews > 0 else 0
            data['mixed_concerns_pct'] = f"{mixed_pct:.1f}"
            
            # Create classification chart data with user-friendly names
            data['classification_distribution'] = {
                'Single Aspect': single_count,
                'Dual Aspect': dual_count,
                'Mixed Concerns': mixed_count
            }
            
            print(f"[DEBUG] Mixed concerns: {mixed_count}/{total_reviews} = {mixed_pct:.1f}%")
            print(f"[DEBUG] Classification: Single={single_count}, Dual={dual_count}, Mixed={mixed_count}")
        
        # =====================================================================
        # LANGUAGE DISTRIBUTION (FIXED)
        # =====================================================================
        language_cols = ['language_detected', 'language', 'lang']
        language_col = None
        for col in language_cols:
            if col in df.columns:
                language_col = col
                break
                
        if language_col:
            lang_counts = df[language_col].value_counts().head(5)
            data['language_distribution'] = lang_counts.to_dict()
            
            print(f"[DEBUG] Language distribution: {dict(lang_counts)}")
        
        # =====================================================================
        # ENSEMBLE METRICS (FIXED)
        # =====================================================================
        ensemble_cols = {
            'method': ['predicted_sentiment_method', 'sentiment_method'],
            'cache': ['predicted_sentiment_from_cache', 'sentiment_from_cache'],
            'models': ['predicted_sentiment_models_used', 'sentiment_models_used']
        }
        
        # Ensemble usage percentage
        method_col = None
        for col in ensemble_cols['method']:
            if col in df.columns:
                method_col = col
                break
                
        if method_col:
            ensemble_count = (df[method_col] == 'two_model_ensemble').sum()
            ensemble_pct = (ensemble_count / total_reviews) * 100 if total_reviews > 0 else 0
            data['ensemble_metrics']['ensemble_usage_pct'] = f"{ensemble_pct:.1f}"
            print(f"[DEBUG] Ensemble usage: {ensemble_count}/{total_reviews} = {ensemble_pct:.1f}%")
        
        # Cache hit rate percentage  
        cache_col = None
        for col in ensemble_cols['cache']:
            if col in df.columns:
                cache_col = col
                break
                
        if cache_col:
            cache_hits = df[cache_col].sum() if df[cache_col].dtype == bool else (df[cache_col] == True).sum()
            cache_pct = (cache_hits / total_reviews) * 100 if total_reviews > 0 else 0
            data['ensemble_metrics']['cache_hit_rate_pct'] = f"{cache_pct:.1f}"
            print(f"[DEBUG] Cache hit rate: {cache_hits}/{total_reviews} = {cache_pct:.1f}%")
        
        # Average models used
        models_col = None
        for col in ensemble_cols['models']:
            if col in df.columns:
                models_col = col
                break
                
        if models_col:
            avg_models = df[models_col].mean() if pd.api.types.is_numeric_dtype(df[models_col]) else 0
            data['ensemble_metrics']['models_used_avg'] = f"{avg_models:.1f}"
            print(f"[DEBUG] Average models used: {avg_models:.1f}")
        
        print(f"[SUCCESS] Generated complete dashboard data for {data['total_reviews']} reviews")
        print(f"[VALIDATION] Final percentages - Mixed: {data['mixed_concerns_pct']}%, UX: {data['ux_priority_pct']}%, High: {data['high_priority_pct']}%")
        
        return data
        
    except Exception as e:
        print(f"[ERROR] Error generating dashboard data: {e}")
        logger.error(f"Error generating dashboard data: {e}")
        return data



# ADDITIONAL DEBUG ROUTE - Add this to your app.py
@app.route('/debug/dashboard-data')  
def debug_dashboard_data():
    """Debug endpoint to see the exact data being sent to frontend"""
    data, demo_mode = get_latest_analysis_data()
    
    return jsonify({
        'demo_mode': demo_mode,
        'raw_data': data,
        'data_keys': list(data.keys()) if data else [],
        'sentiment_distribution': data.get('sentiment_distribution') if data else None,
        'aspect_distribution': data.get('aspect_distribution') if data else None,
        'priority_levels': data.get('priority_levels') if data else None,
        'language_distribution': data.get('language_distribution') if data else None,
        'classification_distribution': data.get('classification_distribution') if data else None,
        'timestamp': datetime.now().isoformat()
    })


# --- ERROR HANDLERS (EXISTING, unchanged) ---

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html', model_status=model_status), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return render_template('500.html', model_status=model_status), 500

@app.errorhandler(413)
def file_too_large(error):
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('upload_file'))

# --- CONTEXT PROCESSORS (EXISTING, unchanged) ---

@app.context_processor
def inject_globals():
    """Make global variables available to all templates"""
    ensemble_info = model_status.get('ensemble_info', {})
    return {
        'model_status': model_status,
        'gpu_available': GPU_AVAILABLE,
        'force_cpu': FORCE_CPU,
        'ensemble_enabled': ensemble_info.get('sentiment_details', {}).get('ensemble_enabled', False),
        'ensemble_models_count': ensemble_info.get('sentiment_details', {}).get('loaded_models', 0),
        'current_year': datetime.now().year
    }

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    print("\n" + "="*70)
    print("MULTILINGUAL SENTIMENT ANALYSIS - ENHANCED PRODUCTION APP")
    print("Two-Model Ensemble Integration + Auto-FedEx Analysis + JSON Fix")
    print("="*70)
    print(f"Project Root: {project_root}")
    print(f"Models Status: {'Loaded' if model_status['loaded'] else 'Not Loaded'}")
    print(f"Device: {model_status.get('device', 'Unknown')}")
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"Force CPU: {FORCE_CPU}")
    print(f"FedEx Scraper: {'Available' if FEDEX_SCRAPER_AVAILABLE else 'Not Available'}")
    print(f"Auto Analysis: {'Enabled' if FEDEX_SCRAPER_AVAILABLE else 'Disabled'}")
    print(f"Dashboard Refresh: Enabled")
    print(f"Debug Mode: {app.config['DEBUG']}")
    print(f"Custom JSON Provider: ACTIVE (handles LineData, timedelta & all custom objects)")
    print(f"Template JSON Safety: ENABLED (pre-processes data for Jinja2)")
    
    # Ensemble information
    ensemble_info = model_status.get('ensemble_info', {})
    if ensemble_info:
        print(f"Pipeline Version: {ensemble_info.get('version', 'Unknown')}")
        sentiment_details = ensemble_info.get('sentiment_details', {})
        if sentiment_details.get('ensemble_enabled'):
            print(f"Ensemble Enabled: YES ({sentiment_details.get('loaded_models', 0)} models)")
        else:
            print(f"Ensemble Enabled: NO")
    
    print("="*70)
    
    print("\nAvailable Endpoints:")
    print("  GET  /              - Homepage with ensemble + FedEx info")
    print("  GET  /analyze       - Text analysis form")
    print("  POST /analyze       - Analyze single text (ensemble)")
    print("  GET  /upload        - File upload form")
    print("  POST /upload        - Process batch file (ensemble + auto-refresh)")
    print("  GET  /dashboard     - Auto-populated BI dashboard")
    print("  GET  /about         - About the project + ensemble details")
    print("  GET  /health        - System health check + FedEx status")
    print("  POST /api/analyze   - REST API single text (ensemble)")
    print("  POST /api/batch     - REST API batch texts (ensemble)")
    print("  GET  /api/dashboard/status  - Dashboard refresh status")
    print("  GET  /api/dashboard/data    - Fresh dashboard data")
    print("  POST /api/fedex/analyze     - Trigger FedEx analysis")
    print("  GET  /export/results/<type> - Export results")
    print("  GET  /download/<filename>   - Download file")
    
    print("\nEnvironment Variables:")
    print("  FORCE_CPU=true     - Force CPU mode (production guarantee)")
    print("  FORCE_GPU=true     - Force GPU mode (when available)")
    print("  FLASK_DEBUG=true   - Enable debug mode")
    print("  SECRET_KEY=<key>   - Set secret key (REQUIRED for production)")
    print("  HF_TOKEN=<token>   - Hugging Face API token (optional)")
    
    print("\nHardware & Deployment Info:")
    device_info = model_status.get('device', 'Unknown')
    if device_info.upper() == 'CPU':
        print("  🖥️  CPU MODE: Full ensemble functionality (18.0 texts/sec)")
        print("     ✅ Production-ready for CPU-only servers")
        print("     ✅ AWS EC2, Google Cloud, Azure compatible")
        print("     ✅ Docker containers (no GPU required)")
        print("     ✅ Lower memory usage, faster loading")
    else:
        print("  🚀 GPU MODE: Full ensemble functionality (16.8 texts/sec)")
        print("     ✅ CUDA acceleration enabled")
        print("     ✅ Higher memory usage, enhanced parallelism")
        print("     ⚠️  GPU hardware required")
    print("     📊 Both modes: XLM-RoBERTa (60%) + Twitter-RoBERTa (40%)")
    
    print("\nEnhanced Features:")
    print("  • Two-model ensemble sentiment analysis")
    print("  • Automatic FedEx review collection & analysis")
    print("  • 7-day data freshness with daily monitoring")
    print("  • Real-time dashboard auto-refresh")
    print("  • Background processing with threading")
    print("  • Cache optimization for repeated queries")
    print("  • CPU/GPU automatic detection & fallback")
    print("  • Professional loading states")
    print("  • Enhanced confidence calibration")
    print("  • FIXED: JSON serialization for all custom objects")
    
    print("\n" + "="*70)
    print("Server starting at http://localhost:5000")
    print("Press CTRL+C to stop")
    print("="*70 + "\n")
    
    # Run the app
    host = '0.0.0.0'  # Allow external connections
    port = int(os.environ.get('PORT', 5000))
    
    app.run(
        host=host,
        port=port,
        debug=app.config['DEBUG'],
        use_reloader=app.config['DEBUG']
    )