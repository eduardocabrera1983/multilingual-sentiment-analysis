import sys
import os
from pathlib import Path

def setup_project_imports():
    """Standardize imports across the entire project"""
    # Get project root (assuming this file is in src/utils/)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    src_path = project_root / 'src'
    
    # Add src to Python path if not already there
    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    
    return {
        'project_root': project_root,
        'src_path': src_path,
        'imports_configured': True
    }

def import_enhanced_models():
    """Safely import enhanced models with fallback"""
    try:
        setup_project_imports()
        from src.models.enhanced_sentiment_classifier import EnhancedSentimentClassifier
        from src.models.enhanced_aspect_classifier import EnhancedAspectClassifier
        return EnhancedSentimentClassifier, EnhancedAspectClassifier, True
    except ImportError as e:
        print(f"⚠️ Enhanced models import failed: {e}")
        return None, None, False

def import_pipeline():
    """Safely import the integrated pipeline"""
    try:
        setup_project_imports()
        from src.integrated_ml_pipeline import IntegratedMLPipeline
        return IntegratedMLPipeline, True
    except ImportError as e:
        print(f"⚠️ Pipeline import failed: {e}")
        return None, False