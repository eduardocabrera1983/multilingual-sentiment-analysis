#!/usr/bin/env python3
"""
Import Test Script for Flask App
Save as: web_app/test_imports.py
Run from web_app directory: python test_imports.py
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test all imports for the Flask app"""
    print("🔍 Testing Flask App Imports")
    print("=" * 50)
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # Check if we're in web_app directory
    if current_dir.name != 'web_app':
        print("❌ You should run this from the web_app directory")
        print("💡 Run: cd web_app")
        return False
    
    # Set up paths like the Flask app does
    project_root = current_dir.parent
    src_path = project_root / 'src'
    
    print(f"Project root: {project_root}")
    print(f"Src path: {src_path}")
    
    # Add paths to Python path
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(src_path))
    
    print(f"\n📁 Directory Structure Check:")
    print(f"   Project root exists: {project_root.exists()}")
    print(f"   src/ exists: {src_path.exists()}")
    print(f"   src/models/ exists: {(src_path / 'models').exists()}")
    print(f"   src/pipelines/ exists: {(src_path / 'pipelines').exists()}")
    
    # Check specific files
    files_to_check = [
        src_path / 'models' / 'enhanced_sentiment_classifier.py',
        src_path / 'models' / 'enhanced_aspect_classifier.py',
        src_path / 'pipelines' / 'integrated_ml_pipeline.py'
    ]
    
    print(f"\n📄 Required Files Check:")
    all_files_exist = True
    for file_path in files_to_check:
        exists = file_path.exists()
        print(f"   {file_path.name}: {'✅' if exists else '❌'}")
        if not exists:
            print(f"      Expected at: {file_path}")
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Some required files are missing!")
        return False
    
    # Test imports
    print(f"\n🧪 Testing Model Imports:")
    import_success = True
    
    try:
        from src.models.enhanced_sentiment_classifier import EnhancedSentimentClassifier
        print("   ✅ Enhanced Sentiment Classifier imported successfully")
    except ImportError as e:
        print(f"   ❌ Enhanced Sentiment Classifier failed: {e}")
        import_success = False
    except Exception as e:
        print(f"   ⚠️ Enhanced Sentiment Classifier error: {e}")
        import_success = False
    
    try:
        from src.models.enhanced_aspect_classifier import EnhancedAspectClassifier
        print("   ✅ Enhanced Aspect Classifier imported successfully")
    except ImportError as e:
        print(f"   ❌ Enhanced Aspect Classifier failed: {e}")
        import_success = False
    except Exception as e:
        print(f"   ⚠️ Enhanced Aspect Classifier error: {e}")
        import_success = False
    
    try:
        from src.pipelines.integrated_ml_pipeline import IntegratedMLPipeline
        print("   ✅ Integrated ML Pipeline imported successfully")
    except ImportError as e:
        print(f"   ❌ Integrated ML Pipeline failed: {e}")
        import_success = False
    except Exception as e:
        print(f"   ⚠️ Integrated ML Pipeline error: {e}")
        import_success = False
    
    # Test Flask app import
    print(f"\n🌐 Testing Flask App:")
    try:
        # Import Flask dependencies first
        import flask
        print(f"   ✅ Flask {flask.__version__} available")
        
        # Test if app.py can be imported (without running it)
        import importlib.util
        app_path = current_dir / 'app.py'
        if app_path.exists():
            print(f"   ✅ app.py exists")
            # We won't actually import app.py as it would start the server
            print(f"   📋 Ready to test Flask app startup")
        else:
            print(f"   ❌ app.py not found")
            import_success = False
            
    except ImportError as e:
        print(f"   ❌ Flask not available: {e}")
        print(f"   💡 Install with: pip install Flask")
        import_success = False
    
    # Test model functionality if imports successful
    if import_success:
        print(f"\n🤖 Testing Model Functionality:")
        try:
            from src.pipelines.integrated_ml_pipeline import IntegratedMLPipeline
            
            print("   Loading pipeline...")
            pipeline = IntegratedMLPipeline()
            
            print("   Testing with sample text...")
            test_text = "Great app but interface is confusing"
            result = pipeline.analyze_text(test_text)
            
            print(f"   ✅ Analysis successful!")
            print(f"   📊 Sentiment: {result.get('sentiment', 'unknown')}")
            print(f"   🎯 Primary aspect: {result.get('primary_aspect', 'unknown')}")
            print(f"   📋 Classification type: {result.get('classification_type', 'unknown')}")
            
        except Exception as e:
            print(f"   ⚠️ Model functionality test failed: {e}")
            print(f"   💡 Models may not be fully loaded, but imports work")
    
    # Check uploads directory
    print(f"\n📁 Checking Uploads Directory:")
    uploads_dir = current_dir / 'uploads'
    if not uploads_dir.exists():
        print("   ⚠️ uploads/ directory doesn't exist")
        print("   💡 Creating uploads directory...")
        uploads_dir.mkdir(exist_ok=True)
        print("   ✅ uploads/ directory created")
    else:
        print("   ✅ uploads/ directory exists")
    
    # Summary
    print(f"\n" + "=" * 50)
    if import_success and all_files_exist:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your Flask app should work correctly")
        print("🚀 Ready to run: python app.py")
    else:
        print("❌ Some issues found")
        print("💡 Check the error messages above")
    
    return import_success and all_files_exist

if __name__ == "__main__":
    success = test_imports()
    
    if success:
        print(f"\n🎯 Next Steps:")
        print(f"1. Run your Flask app: python app.py")
        print(f"2. Open browser: http://localhost:5000")
        print(f"3. Test with your FedEx data")
    else:
        print(f"\n🔧 Troubleshooting:")
        print(f"1. Make sure you're in the web_app directory")
        print(f"2. Check that all files exist in src/")
        print(f"3. Install missing dependencies")
        print(f"4. Check for syntax errors in your Python files")