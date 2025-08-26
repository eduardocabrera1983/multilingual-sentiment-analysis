#!/usr/bin/env python3
"""
Complete Flask App Setup Script
This will create all missing templates and static files for your Flask app
Run this from your project root: multilingual-sentiment-analysis/
"""

import os
from pathlib import Path

def create_flask_structure():
    """Create all necessary Flask app files and directories"""
    
    print("\n" + "="*70)
    print("SETTING UP FLASK APP STRUCTURE")
    print("="*70)
    
    # Get project root
    project_root = Path.cwd()
    print(f"Project root: {project_root}")
    
    # Create directory structure
    directories = [
        'templates',
        'static',
        'static/css',
        'static/js',
        'uploads',
        'cache',
        '.cache',
        'web_app/uploads'
    ]
    
    print("\n📁 Creating directories...")
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    # Create base.html template
    print("\n📄 Creating templates...")
    
    base_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Multilingual Sentiment Analysis{% endblock %}</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Bootstrap Icons -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css" rel="stylesheet">
    <!-- Custom CSS -->
    <link href="{{ url_for('static', filename='css/style.css') }}" rel="stylesheet">
    
    {% block extra_head %}{% endblock %}
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary sticky-top">
        <div class="container">
            <a class="navbar-brand fw-bold" href="{{ url_for('index') }}">
                <i class="bi bi-chat-text-fill me-2"></i>
                ML Sentiment Analysis
            </a>
            
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav me-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('index') }}">
                            <i class="bi bi-house-fill me-1"></i>Home
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{{ url_for('analyze_text') }}">
                            <i class="bi bi-search me-1"></i>Analyze
                        </a>
                    </li>
                </ul>
                
                <!-- Model Status Indicator -->
                <div class="navbar-nav">
                    {% if model_status and model_status.loaded %}
                        <span class="nav-link text-success">
                            <i class="bi bi-check-circle-fill me-1"></i>Models Active
                        </span>
                    {% else %}
                        <span class="nav-link text-warning">
                            <i class="bi bi-exclamation-triangle-fill me-1"></i>Basic Mode
                        </span>
                    {% endif %}
                </div>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <main>
        {% block content %}{% endblock %}
    </main>

    <!-- Footer -->
    <footer class="bg-dark text-light py-4 mt-5">
        <div class="container text-center">
            <p>Multilingual Sentiment Analysis - Eduardo Cabrera</p>
        </div>
    </footer>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    
    {% block extra_scripts %}{% endblock %}
</body>
</html>'''
    
    (project_root / 'templates' / 'base.html').write_text(base_html, encoding='utf-8')
    print("  ✅ Created: base.html")
    
    # Create index.html
    index_html = '''{% extends "base.html" %}

{% block title %}Home - Multilingual Sentiment Analysis{% endblock %}

{% block content %}
<div class="container my-5">
    <div class="row">
        <div class="col-lg-8 mx-auto text-center">
            <h1 class="display-4 mb-4">Multilingual Sentiment Analysis</h1>
            <p class="lead mb-4">
                Advanced ML system with multi-label aspect classification, 
                user experience prioritization, and business intelligence generation.
            </p>
            
            <div class="row g-3 justify-content-center">
                <div class="col-md-4">
                    <div class="card h-100">
                        <div class="card-body">
                            <i class="bi bi-search display-4 text-primary mb-3"></i>
                            <h5 class="card-title">Text Analysis</h5>
                            <p class="card-text">Analyze individual texts with detailed insights</p>
                            <a href="{{ url_for('analyze_text') }}" class="btn btn-primary">
                                Start Analysis
                            </a>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-4">
                    <div class="card h-100">
                        <div class="card-body">
                            <i class="bi bi-cloud-upload display-4 text-success mb-3"></i>
                            <h5 class="card-title">Batch Processing</h5>
                            <p class="card-text">Upload CSV files for bulk analysis</p>
                            <a href="{{ url_for('analyze_text') }}" class="btn btn-success">
                                Upload File
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- System Status -->
    <div class="row mt-5">
        <div class="col-lg-8 mx-auto">
            <div class="card">
                <div class="card-header bg-primary text-white">
                    <h5 class="mb-0"><i class="bi bi-cpu me-2"></i>System Status</h5>
                </div>
                <div class="card-body">
                    {% if model_status and model_status.loaded %}
                        <div class="alert alert-success">
                            <i class="bi bi-check-circle-fill me-2"></i>
                            <strong>Enhanced Models Active</strong>
                            <ul class="mb-0 mt-2">
                                <li>XLM-RoBERTa Sentiment Analysis</li>
                                <li>Multi-Label Aspect Classification</li>
                                <li>User Experience Prioritization</li>
                                <li>Business Intelligence Generation</li>
                            </ul>
                        </div>
                    {% else %}
                        <div class="alert alert-warning">
                            <i class="bi bi-exclamation-triangle-fill me-2"></i>
                            <strong>Basic Mode Active</strong>
                            <p class="mb-0 mt-2">Enhanced models not loaded. Basic analysis available.</p>
                            {% if model_status and model_status.error %}
                                <small class="text-muted">Error: {{ model_status.error }}</small>
                            {% endif %}
                        </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
    
    (project_root / 'templates' / 'index.html').write_text(index_html, encoding='utf-8')
    print("  ✅ Created: index.html")
    
    # Create analyze.html
    analyze_html = '''{% extends "base.html" %}

{% block title %}Analyze Text - Multilingual Sentiment Analysis{% endblock %}

{% block content %}
<div class="container my-5">
    <div class="row">
        <div class="col-lg-8 mx-auto">
            <h1 class="mb-4">Text Analysis</h1>
            
            <div class="card">
                <div class="card-body">
                    <form method="POST" action="{{ url_for('analyze_text') }}">
                        <div class="mb-3">
                            <label for="text" class="form-label">Text to Analyze</label>
                            <textarea class="form-control" id="text" name="text" 
                                      rows="6" required placeholder="Enter your text here..."></textarea>
                        </div>
                        
                        <div class="mb-3">
                            <label for="language" class="form-label">Language (Optional)</label>
                            <select class="form-select" id="language" name="language">
                                <option value="auto">Auto-detect</option>
                                <option value="en">English</option>
                                <option value="es">Spanish</option>
                                <option value="de">German</option>
                                <option value="fr">French</option>
                                <option value="nl">Dutch</option>
                            </select>
                        </div>
                        
                        <button type="submit" class="btn btn-primary btn-lg">
                            <i class="bi bi-cpu me-2"></i>Analyze Text
                        </button>
                    </form>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
    
    (project_root / 'templates' / 'analyze.html').write_text(analyze_html, encoding='utf-8')
    print("  ✅ Created: analyze.html")
    
    # Create results.html
    results_html = '''{% extends "base.html" %}

{% block title %}Analysis Results - Multilingual Sentiment Analysis{% endblock %}

{% block content %}
<div class="container my-4">
    <h1 class="mb-4">Analysis Results</h1>
    
    {% if result %}
    <div class="row">
        <div class="col-lg-8">
            <!-- Original Text -->
            <div class="card mb-4">
                <div class="card-header">
                    <h5 class="mb-0"><i class="bi bi-chat-quote me-2"></i>Analyzed Text</h5>
                </div>
                <div class="card-body">
                    <p class="mb-0">{{ text }}</p>
                </div>
            </div>
            
            <!-- Sentiment Analysis -->
            <div class="card mb-4">
                <div class="card-header">
                    <h5 class="mb-0"><i class="bi bi-heart-pulse me-2"></i>Sentiment Analysis</h5>
                </div>
                <div class="card-body">
                    <div class="row align-items-center">
                        <div class="col-md-6 text-center">
                            {% set sentiment_class = 'success' if result.sentiment == 'positive' else 'danger' if result.sentiment == 'negative' else 'secondary' %}
                            <h3>
                                <span class="badge bg-{{ sentiment_class }}">
                                    {{ result.sentiment|upper }}
                                </span>
                            </h3>
                        </div>
                        <div class="col-md-6">
                            <p class="mb-1"><strong>Confidence:</strong> {{ "%.1f"|format(result.sentiment_confidence * 100) }}%</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Multi-Label Aspects -->
            <div class="card mb-4">
                <div class="card-header">
                    <h5 class="mb-0"><i class="bi bi-diagram-3 me-2"></i>Multi-Label Aspect Classification</h5>
                </div>
                <div class="card-body">
                    <p><strong>Primary Aspect:</strong> 
                        <span class="badge bg-primary">{{ result.primary_aspect.replace('_', ' ').title() }}</span>
                    </p>
                    {% if result.secondary_aspects %}
                    <p><strong>Secondary Aspects:</strong>
                        {% for aspect in result.secondary_aspects %}
                        <span class="badge bg-warning">{{ aspect.replace('_', ' ').title() }}</span>
                        {% endfor %}
                    </p>
                    {% endif %}
                    <p><strong>Classification Type:</strong> 
                        <span class="badge bg-info">{{ result.classification_type.replace('_', ' ').title() }}</span>
                    </p>
                </div>
            </div>
        </div>
        
        <div class="col-lg-4">
            <!-- Business Intelligence -->
            <div class="card">
                <div class="card-header bg-primary text-white">
                    <h5 class="mb-0"><i class="bi bi-lightbulb me-2"></i>Business Intelligence</h5>
                </div>
                <div class="card-body">
                    <p><strong>Priority Level:</strong> 
                        {% set priority_class = 'danger' if result.priority_level == 'HIGH' else 'warning' if result.priority_level == 'MEDIUM' else 'success' %}
                        <span class="badge bg-{{ priority_class }}">{{ result.priority_level }}</span>
                    </p>
                    <p><strong>Summary:</strong><br>{{ result.business_summary }}</p>
                    <p><strong>Recommendation:</strong><br>{{ result.recommendation }}</p>
                </div>
            </div>
        </div>
    </div>
    
    <div class="mt-4">
        <a href="{{ url_for('analyze_text') }}" class="btn btn-primary">
            <i class="bi bi-arrow-left me-2"></i>Analyze Another Text
        </a>
    </div>
    {% endif %}
</div>
{% endblock %}'''
    
    (project_root / 'templates' / 'results.html').write_text(results_html, encoding='utf-8')
    print("  ✅ Created: results.html")
    
    # Create simple versions of other templates
    templates = {
        'upload.html': '''{% extends "base.html" %}
{% block title %}Upload File{% endblock %}
{% block content %}
<div class="container my-5">
    <h1>Upload CSV File</h1>
    <p>File upload functionality coming soon...</p>
</div>
{% endblock %}''',
        
        'batch_results.html': '''{% extends "base.html" %}
{% block title %}Batch Results{% endblock %}
{% block content %}
<div class="container my-5">
    <h1>Batch Analysis Results</h1>
    <p>Batch results will appear here...</p>
</div>
{% endblock %}''',
        
        'dashboard.html': '''{% extends "base.html" %}
{% block title %}Dashboard{% endblock %}
{% block content %}
<div class="container my-5">
    <h1>Business Intelligence Dashboard</h1>
    <p>Dashboard functionality coming soon...</p>
</div>
{% endblock %}''',
        
        'about.html': '''{% extends "base.html" %}
{% block title %}About{% endblock %}
{% block content %}
<div class="container my-5">
    <h1>About This Project</h1>
    <p class="lead">Advanced multilingual sentiment analysis with multi-label classification.</p>
    <h3>Features:</h3>
    <ul>
        <li>Multi-label aspect classification</li>
        <li>User experience prioritization</li>
        <li>Business intelligence generation</li>
        <li>Support for 5+ languages</li>
    </ul>
    <p><strong>Developer:</strong> Eduardo Cabrera</p>
    <p><strong>Date:</strong> August 2025</p>
</div>
{% endblock %}''',
        
        '404.html': '''{% extends "base.html" %}
{% block title %}Page Not Found{% endblock %}
{% block content %}
<div class="container my-5 text-center">
    <h1 class="display-1">404</h1>
    <p class="lead">Page not found</p>
    <a href="{{ url_for('index') }}" class="btn btn-primary">Go Home</a>
</div>
{% endblock %}''',
        
        '500.html': '''{% extends "base.html" %}
{% block title %}Server Error{% endblock %}
{% block content %}
<div class="container my-5 text-center">
    <h1 class="display-1">500</h1>
    <p class="lead">Server Error</p>
    <a href="{{ url_for('index') }}" class="btn btn-primary">Go Home</a>
</div>
{% endblock %}'''
    }
    
    for filename, content in templates.items():
        (project_root / 'templates' / filename).write_text(content, encoding='utf-8')
        print(f"  ✅ Created: {filename}")
    
    # Create CSS file
    print("\n📄 Creating static files...")
    
    css_content = '''/* Custom CSS for Multilingual Sentiment Analysis */

:root {
    --primary-color: #0d6efd;
    --success-color: #198754;
    --warning-color: #ffc107;
    --danger-color: #dc3545;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.hero-section {
    background: linear-gradient(135deg, #0d6efd 0%, #6610f2 100%);
    color: white;
    padding: 4rem 0;
}

.card {
    border: none;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
}

.badge {
    padding: 0.5rem 1rem;
    font-size: 0.9rem;
}

footer {
    margin-top: auto;
}'''
    
    (project_root / 'static' / 'css' / 'style.css').write_text(css_content, encoding='utf-8')
    print("  ✅ Created: style.css")
    
    # Create JS file
    js_content = '''// Custom JavaScript for Multilingual Sentiment Analysis

document.addEventListener('DOMContentLoaded', function() {
    console.log('ML Sentiment Analysis App Ready');
    
    // Add any custom JavaScript here
});'''
    
    (project_root / 'static' / 'js' / 'main.js').write_text(js_content, encoding='utf-8')
    print("  ✅ Created: main.js")
    
    # Create simplified app.py if it doesn't exist
    app_py_path = project_root / 'app.py'
    if not app_py_path.exists():
        print("\n📄 Creating simplified app.py...")
        
        app_py = '''#!/usr/bin/env python3
"""
Flask App for Multilingual Sentiment Analysis - Simplified Version
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Fix import paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Import models
try:
    from src.models.enhanced_sentiment_classifier import EnhancedSentimentClassifier
    from src.models.enhanced_aspect_classifier import EnhancedAspectClassifier
    from src.integrated_ml_pipeline import IntegratedMLPipeline
    MODELS_AVAILABLE = True
    print("[SUCCESS] Models imported successfully!")
except ImportError as e:
    print(f"[WARNING] Could not import models: {e}")
    MODELS_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'multilingual-sentiment-2025'

# Initialize models
ml_pipeline = None
model_status = {'loaded': False, 'error': None}

def load_models():
    global ml_pipeline, model_status
    try:
        if MODELS_AVAILABLE:
            ml_pipeline = IntegratedMLPipeline()
            model_status = {'loaded': True, 'error': None}
            print("[SUCCESS] ML Pipeline loaded!")
    except Exception as e:
        model_status = {'loaded': False, 'error': str(e)}
        print(f"[ERROR] Failed to load models: {e}")

# Load models on startup
load_models()

@app.route('/')
def index():
    return render_template('index.html', model_status=model_status)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze_text():
    if request.method == 'POST':
        text = request.form.get('text', '')
        language = request.form.get('language', 'auto')
        
        if ml_pipeline and model_status['loaded']:
            result = ml_pipeline.analyze_text(text, language)
        else:
            # Fallback result
            result = {
                'sentiment': 'neutral',
                'sentiment_confidence': 0.5,
                'primary_aspect': 'general_satisfaction',
                'secondary_aspects': [],
                'classification_type': 'single_aspect',
                'priority_level': 'MEDIUM',
                'business_summary': 'Basic analysis - models not loaded',
                'recommendation': 'Please load models for full analysis'
            }
        
        return render_template('results.html', text=text, result=result, model_status=model_status)
    
    return render_template('analyze.html', model_status=model_status)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'models_loaded': model_status['loaded']})

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html', model_status=model_status), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html', model_status=model_status), 500

if __name__ == '__main__':
    print("Starting Flask app on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
        
        app_py_path.write_text(app_py, encoding='utf-8')
        print("  ✅ Created: app.py")
    
    print("\n" + "="*70)
    print("✅ FLASK APP SETUP COMPLETE!")
    print("="*70)
    
    print("\nNext steps:")
    print("1. Make sure your models are in src/models/")
    print("2. Make sure integrated_ml_pipeline.py is in src/")
    print("3. Install Flask if needed: pip install flask")
    print("4. Run the app: python app.py")
    print("5. Open browser to: http://localhost:5000")
    
    return True


if __name__ == "__main__":
    success = create_flask_structure()
    if success:
        print("\n🎉 Your Flask app is ready to run!")
        print("Run: python app.py")
    else:
        print("\n❌ Setup failed. Please check errors above.")