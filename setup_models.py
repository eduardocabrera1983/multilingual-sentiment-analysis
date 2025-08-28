#!/usr/bin/env python3
"""
FIXED Model Setup and Initialization Script
Ensures all models are properly set up and ready to use
Run this from anywhere in your project to set up the environment
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import json
import shutil

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(text: str, color: str = Colors.ENDC):
    """Print colored text to terminal"""
    print(f"{color}{text}{Colors.ENDC}")

def print_section(title: str):
    """Print a formatted section header"""
    print_colored(f"\n{'='*70}", Colors.CYAN)
    print_colored(f"🔧 {title}", Colors.BOLD)
    print_colored(f"{'='*70}", Colors.CYAN)

def find_project_root():
    """Find the actual project root regardless of where script is run from"""
    current = Path(__file__).parent.resolve()
    
    # Look for key indicators of project root
    indicators = ['src', 'web_app', 'data', '.gitignore', 'README.md']
    
    # Check current directory and up to 3 levels up
    for _ in range(4):
        if any((current / indicator).exists() for indicator in indicators):
            # Verify this looks like our project structure
            if (current / 'src' / 'models').exists() or (current / 'src').exists():
                return current
        current = current.parent
    
    # Fallback: use directory where script is located
    return Path(__file__).parent.resolve()

class ModelSetup:
    def __init__(self):
        # FIXED: Find actual project root
        self.project_root = find_project_root()
        self.src_path = self.project_root / 'src'
        self.models_path = self.src_path / 'models'
        self.scrapers_path = self.src_path / 'scrapers'  # Added scrapers path
        self.data_path = self.project_root / 'data'
        self.test_results_path = self.project_root / 'test_results'
        
        print_colored(f"🎯 Detected project root: {self.project_root}", Colors.BLUE)
        
    def check_python_version(self):
        """Check if Python version is 3.7 or higher"""
        print_section("Python Version Check")
        
        import sys
        version = sys.version_info
        
        if version.major == 3 and version.minor >= 7:
            print_colored(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible", Colors.GREEN)
            return True
        else:
            print_colored(f"❌ Python {version.major}.{version.minor} - Requires Python 3.7+", Colors.FAIL)
            return False
    
    def create_directory_structure(self):
        """Create necessary directories"""
        print_section("Creating Directory Structure")
        
        directories = [
            self.src_path,
            self.models_path,
            self.scrapers_path,  # Added scrapers directory
            self.data_path,
            self.test_results_path,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created/Verified: {directory}")
        
        # Create __init__.py files
        init_files = [
            self.src_path / '__init__.py',
            self.models_path / '__init__.py',
            self.scrapers_path / '__init__.py',  # Added scrapers __init__.py
        ]
        
        for init_file in init_files:
            if not init_file.exists():
                init_file.write_text('# ML Models Package\n')
                print(f"📄 Created: {init_file}")
            else:
                print(f"✅ Exists: {init_file}")
        
        return True
    
    def check_and_install_dependencies(self):
        """Check and install required dependencies"""
        print_section("Checking Dependencies")
        
        required_packages = {
            'pandas': 'pandas>=1.3.0',
            'numpy': 'numpy>=1.21.0',
            'sklearn': 'scikit-learn>=1.0.0',  # FIXED: Import as 'sklearn' but install as 'scikit-learn'
            'torch': 'torch>=1.9.0',
            'transformers': 'transformers>=4.20.0',
            'langdetect': 'langdetect',
            'streamlit': 'streamlit>=1.10.0',
            'plotly': 'plotly>=5.0.0',
            'google_play_scraper': 'google-play-scraper',  # FIXED: Import uses underscores
            'flask': 'flask>=2.0.0',
        }
        
        missing_packages = []
        
        for import_name, install_name in required_packages.items():
            try:
                # Handle special import cases
                if import_name == 'google_play_scraper':
                    __import__('google_play_scraper')
                else:
                    __import__(import_name)
                print(f"✅ {install_name.split('>=')[0]} installed")
            except ImportError:
                print(f"❌ {install_name.split('>=')[0]} not installed")
                missing_packages.append(install_name)
        
        if missing_packages:
            print_colored(f"\n📦 Installing missing packages...", Colors.WARNING)
            for package in missing_packages:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print_colored("✅ All dependencies installed!", Colors.GREEN)
        else:
            print_colored("✅ All dependencies already installed!", Colors.GREEN)
        
        return True
    
    def verify_model_files(self):
        """Verify that model files exist - CORRECTED PATHS"""
        print_section("Verifying Model Files")
        
        # FIXED: Use correct paths
        model_files = {
            'enhanced_sentiment_classifier.py': self.models_path / 'enhanced_sentiment_classifier.py',
            'enhanced_aspect_classifier.py': self.models_path / 'enhanced_aspect_classifier.py',
            'integrated_ml_pipeline.py': self.src_path / 'integrated_ml_pipeline.py',  # FIXED: src/ not src/pipelines/
            'fedex_scraper.py': self.src_path / 'scrapers' / 'fedex_scraper.py',  # FIXED: src/scrapers/ location
        }
        
        all_exist = True
        
        for name, path in model_files.items():
            if path.exists():
                print(f"✅ Found: {name} at {path}")
            else:
                print_colored(f"❌ Missing: {name} at {path}", Colors.FAIL)
                all_exist = False
        
        if not all_exist:
            print_colored("\n⚠️ Some model files are missing!", Colors.WARNING)
            print("Please ensure all model files are in the correct locations.")
            print("Expected structure:")
            print("  src/models/enhanced_sentiment_classifier.py")
            print("  src/models/enhanced_aspect_classifier.py") 
            print("  src/integrated_ml_pipeline.py")
            print("  src/scrapers/fedex_scraper.py")  # FIXED: correct scraper path
        else:
            print_colored("✅ All model files found!", Colors.GREEN)
        
        return all_exist
    
    def download_transformer_models(self):
        """Download required transformer models"""
        print_section("Downloading Transformer Models")
        
        models_to_download = [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "nlptown/bert-base-multilingual-uncased-sentiment", 
            "microsoft/DialoGPT-medium",
        ]
        
        failed_downloads = []
        
        for model_name in models_to_download:
            try:
                print(f"📥 Downloading {model_name}...")
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                print(f"✅ Downloaded: {model_name}")
                
                # Clean up memory
                del tokenizer, model
                
            except Exception as e:
                print_colored(f"❌ Failed to download {model_name}: {e}", Colors.FAIL)
                failed_downloads.append(model_name)
        
        if failed_downloads:
            print_colored(f"\n⚠️ Failed to download {len(failed_downloads)} model(s)", Colors.WARNING)
            for model in failed_downloads:
                print(f"   - {model}")
            print("Models will be downloaded automatically when first used.")
        else:
            print_colored("\n✅ All transformer models downloaded successfully!", Colors.GREEN)
        
        return len(failed_downloads) < len(models_to_download)
    
    def test_model_imports(self):
        """Test if models can be imported"""
        print_section("Testing Model Imports")
        
        # Add paths to Python path
        sys.path.insert(0, str(self.project_root))
        sys.path.insert(0, str(self.src_path))
        
        import_success = True
        
        # Test Enhanced Sentiment Classifier
        try:
            from src.models.enhanced_sentiment_classifier import EnhancedSentimentClassifier
            print("✅ Enhanced Sentiment Classifier imports successfully")
        except Exception as e:
            print_colored(f"❌ Enhanced Sentiment Classifier import failed: {e}", Colors.FAIL)
            import_success = False
        
        # Test Enhanced Aspect Classifier
        try:
            from src.models.enhanced_aspect_classifier import EnhancedAspectClassifier
            print("✅ Enhanced Aspect Classifier imports successfully")
        except Exception as e:
            print_colored(f"❌ Enhanced Aspect Classifier import failed: {e}", Colors.FAIL)
            import_success = False
        
        # Test Integrated Pipeline - FIXED PATH
        try:
            from src.integrated_ml_pipeline import IntegratedMLPipeline
            print("✅ Integrated ML Pipeline imports successfully")
        except Exception as e:
            print_colored(f"❌ Integrated ML Pipeline import failed: {e}", Colors.FAIL)
            import_success = False
        
        return import_success
    
    def test_basic_functionality(self):
        """Test basic model functionality"""
        print_section("Testing Basic Functionality")
        
        try:
            # Add paths
            sys.path.insert(0, str(self.project_root))
            sys.path.insert(0, str(self.src_path))
            
            from src.models.enhanced_sentiment_classifier import EnhancedSentimentClassifier
            from src.models.enhanced_aspect_classifier import EnhancedAspectClassifier
            
            print("Initializing models...")
            
            # Test sentiment classifier
            print("\n📊 Testing Sentiment Classifier...")
            sentiment_model = EnhancedSentimentClassifier(use_ensemble=True)
            test_result = sentiment_model.analyze_sentiment("This is a great product!")
            print(f"✅ Sentiment Analysis: {test_result['sentiment']} ({test_result['confidence']:.1%})")
            
            # Test aspect classifier  
            print("\n🎯 Testing Aspect Classifier...")
            aspect_model = EnhancedAspectClassifier()
            aspect_result = aspect_model.classify_aspects_multilabel(
                text="Great app but interface is confusing",
                language='en',
                sentiment='negative',
                sentiment_confidence=0.7
            )
            print(f"✅ Aspect Classification: {aspect_result.get('primary_aspect', 'detected')}")
            print(f"   Classification Type: {aspect_result.get('classification_type', 'unknown')}")
            print(f"   Priority Level: {aspect_result.get('priority_level', 'unknown')}")
            
            print_colored("\n✅ All basic functionality tests passed!", Colors.GREEN)
            return True
            
        except Exception as e:
            print_colored(f"❌ Functionality test failed: {e}", Colors.FAIL)
            return False
    
    def create_requirements_file(self):
        """Create consolidated requirements.txt file"""
        print_section("Creating Requirements File")
        
        # Use the comprehensive requirements content
        requirements_content = """# MULTILINGUAL SENTIMENT ANALYSIS - CONSOLIDATED REQUIREMENTS
# Install with: pip install -r requirements.txt

# Core Data Science & ML
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
scipy>=1.8.0

# Deep Learning & Transformers
torch>=1.9.0
transformers>=4.30.0
tokenizers>=0.13.0

# Natural Language Processing
langdetect>=1.0.9

# Web Framework & API
flask>=2.3.0
werkzeug>=2.3.0
gunicorn>=20.1.0
requests>=2.28.0

# Data Visualization
plotly>=5.0.0
streamlit>=1.25.0

# Data Collection
google-play-scraper>=3.0.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.64.0
joblib>=1.3.0
"""
        
        requirements_path = self.project_root / 'requirements.txt'
        requirements_path.write_text(requirements_content.strip())
        print(f"✅ Created consolidated: {requirements_path}")
        
        # Remove any old requirements files to avoid confusion
        old_locations = [
            self.project_root / 'web_app' / 'requirements.txt',
            self.project_root / 'src' / 'requirements.txt',
            self.project_root / 'tests' / 'requirements.txt'
        ]
        
        for old_req in old_locations:
            if old_req.exists():
                try:
                    old_req.unlink()
                    print(f"🗑️ Removed old: {old_req}")
                except:
                    print(f"⚠️ Could not remove: {old_req}")
        
        return True
    
    def create_sample_config(self):
        """Create sample configuration file"""
        print_section("Creating Configuration File")
        
        config = {
            "model_settings": {
                "use_gpu": True,
                "confidence_threshold": 0.7,
                "max_sequence_length": 512
            },
            "paths": {
                "models_dir": "src/models",
                "data_dir": "data", 
                "cache_dir": ".cache"
            },
            "api_settings": {
                "host": "0.0.0.0",
                "port": 5000,
                "debug": False
            }
        }
        
        config_path = self.project_root / 'config.json'
        config_path.write_text(json.dumps(config, indent=2))
        print(f"✅ Created: {config_path}")
        return True
    
    def print_setup_summary(self, results):
        """Print setup summary"""
        print_section("SETUP SUMMARY")
        
        print_colored("\n📋 Setup Results:", Colors.BOLD)
        for step, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {step}")
        
        if all(results.values()):
            print_colored("\n🎉 SETUP COMPLETE!", Colors.GREEN + Colors.BOLD)
            
            print_colored("\n📚 Next Steps:", Colors.CYAN)
            print("1. Test the models:")
            print("   python -c \"from src.integrated_ml_pipeline import IntegratedMLPipeline; p=IntegratedMLPipeline(); print(p.analyze_text('Great app!'))\"")
            print("\n2. Run the Flask app:")
            print("   python app.py")
            print("\n3. Test the web interface:")
            print("   cd web_app && python test_imports.py")
            print("\n4. Analyze your data:")
            print("   python src/scrapers/fedex_scraper.py")  # FIXED: correct scraper path
            
            print_colored("\n🚀 Your ML system is ready!", Colors.GREEN)
            
        else:
            print_colored("\n⚠️ SETUP INCOMPLETE", Colors.WARNING)
            print("Some steps failed. Please review the errors above.")
            print("\nTroubleshooting:")
            print("1. Ensure you have Python 3.7+")
            print("2. Check internet connection for model downloads")
            print("3. Verify all files from repository are present")
            print("4. Try manual installation: pip install -r requirements.txt")
    
    def run_setup(self):
        """Run complete setup process"""
        print_colored("\n" + "="*70, Colors.BOLD)
        print_colored("🚀 MULTILINGUAL SENTIMENT ANALYSIS - MODEL SETUP", Colors.BOLD)
        print_colored("="*70, Colors.BOLD)
        print("Setting up enhanced ML models with multi-label classification...")
        
        results = {}
        
        # Run setup steps
        results['Python Version'] = self.check_python_version()
        
        if results['Python Version']:
            results['Directory Structure'] = self.create_directory_structure()
            results['Dependencies'] = self.check_and_install_dependencies()
            results['Model Files'] = self.verify_model_files()
            
            if results['Model Files']:
                results['Model Imports'] = self.test_model_imports()
                results['Transformer Models'] = self.download_transformer_models()
                
                if results['Model Imports']:
                    results['Basic Functionality'] = self.test_basic_functionality()
            
            results['Requirements File'] = self.create_requirements_file()
            results['Configuration'] = self.create_sample_config()
        
        # Print summary
        self.print_setup_summary(results)
        
        return all(results.values())


def main():
    """Main setup function"""
    setup = ModelSetup()
    success = setup.run_setup()
    
    if success:
        print_colored("\n✅ Setup complete! Your ML models are ready to use.", Colors.GREEN + Colors.BOLD)
        print_colored("Run your Flask app with: python app.py", Colors.CYAN)
    else:
        print_colored("\n⚠️ Setup completed with some issues. Check messages above.", Colors.WARNING)
    
    return success


if __name__ == "__main__":
    main()