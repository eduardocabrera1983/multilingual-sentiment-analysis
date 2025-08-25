#!/usr/bin/env python3
"""
Model Setup and Initialization Script
Ensures all models are properly set up and ready to use
Run this after cloning the repository to set up the environment
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

class ModelSetup:
    def __init__(self):
        self.project_root = Path.cwd()
        self.src_path = self.project_root / 'src'
        self.models_path = self.src_path / 'models'
        self.data_path = self.project_root / 'data'
        self.test_results_path = self.project_root / 'test_results'
        
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
            'scikit-learn': 'scikit-learn>=0.24.0',
            'torch': 'torch>=1.9.0',
            'transformers': 'transformers>=4.20.0',
            'langdetect': 'langdetect',
            'streamlit': 'streamlit>=1.10.0',
            'plotly': 'plotly>=5.0.0',
            'google-play-scraper': 'google-play-scraper',
        }
        
        missing_packages = []
        
        for package_name, install_name in required_packages.items():
            try:
                __import__(package_name.replace('-', '_').split('>')[0])
                print(f"✅ {package_name} installed")
            except ImportError:
                print(f"❌ {package_name} not installed")
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
        """Verify that model files exist"""
        print_section("Verifying Model Files")
        
        model_files = {
            'enhanced_sentiment_classifier.py': self.models_path / 'enhanced_sentiment_classifier.py',
            'enhanced_aspect_classifier.py': self.models_path / 'enhanced_aspect_classifier.py',
            'integrated_ml_pipeline.py': self.src_path / 'integrated_ml_pipeline.py',
            'fedex_scraper.py': self.src_path / 'fedex_scraper.py',
        }
        
        all_exist = True
        
        for name, path in model_files.items():
            if path.exists():
                print(f"✅ Found: {name}")
            else:
                print_colored(f"❌ Missing: {name} at {path}", Colors.FAIL)
                all_exist = False
        
        if not all_exist:
            print_colored("\n⚠️ Some model files are missing!", Colors.WARNING)
            print("Please ensure all model files are in the correct locations.")
            print("Check that you've cloned the complete repository.")
            return False
        
        return True
    
    def download_transformer_models(self):
        """Pre-download transformer models to cache"""
        print_section("Downloading Transformer Models")
        print("This may take a few minutes on first run...")
        
        models_to_download = [
            {
                'name': 'XLM-RoBERTa Sentiment',
                'model_id': 'cardiffnlp/twitter-xlm-roberta-base-sentiment',
                'type': 'sentiment'
            },
            {
                'name': 'Multilingual BERT Sentiment',
                'model_id': 'nlptown/bert-base-multilingual-uncased-sentiment',
                'type': 'sentiment'
            },
            {
                'name': 'DistilBERT Multilingual',
                'model_id': 'lxyuan/distilbert-base-multilingual-cased-sentiments-student',
                'type': 'sentiment'
            },
            {
                'name': 'BART Zero-Shot',
                'model_id': 'facebook/bart-large-mnli',
                'type': 'zero-shot'
            }
        ]
        
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        
        failed_downloads = []
        
        for model_info in models_to_download:
            try:
                print(f"\n📥 Downloading {model_info['name']}...")
                
                if model_info['type'] == 'sentiment':
                    # Download sentiment analysis model
                    pipeline('sentiment-analysis', model=model_info['model_id'])
                elif model_info['type'] == 'zero-shot':
                    # Download zero-shot classification model
                    pipeline('zero-shot-classification', model=model_info['model_id'])
                
                print(f"✅ {model_info['name']} ready")
                
            except Exception as e:
                print_colored(f"⚠️ Failed to download {model_info['name']}: {e}", Colors.WARNING)
                failed_downloads.append(model_info['name'])
        
        if failed_downloads:
            print_colored(f"\n⚠️ Some models failed to download: {failed_downloads}", Colors.WARNING)
            print("The system will still work but may have reduced functionality.")
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
        
        # Test Integrated Pipeline
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
            print(f"✅ Sentiment: {test_result['sentiment']} (confidence: {test_result['confidence']:.2f})")
            
            # Test aspect classifier
            print("\n🎯 Testing Aspect Classifier...")
            aspect_model = EnhancedAspectClassifier(confidence_threshold=0.3)
            test_result = aspect_model.classify_aspects_multilabel("The interface is confusing but tracking works well")
            print(f"✅ Primary Aspect: {test_result['primary_aspect']}")
            print(f"   Secondary Aspects: {test_result['secondary_aspects']}")
            print(f"   Classification Type: {test_result['classification_type']}")
            
            print_colored("\n✅ Basic functionality test passed!", Colors.GREEN)
            return True
            
        except Exception as e:
            print_colored(f"❌ Functionality test failed: {e}", Colors.FAIL)
            return False
    
    def create_requirements_file(self):
        """Create requirements.txt file"""
        print_section("Creating Requirements File")
        
        requirements = """# Core Dependencies
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0

# Deep Learning
torch>=1.9.0
transformers>=4.20.0

# NLP and Language Detection
langdetect

# Web Interface
streamlit>=1.10.0

# Visualization
plotly>=5.0.0
matplotlib>=3.3.0
seaborn>=0.11.0

# Data Collection
google-play-scraper

# Optional but Recommended
jupyter>=1.0.0
ipykernel>=6.0.0
openpyxl>=3.0.0
"""
        
        req_file = self.project_root / 'requirements.txt'
        req_file.write_text(requirements)
        print(f"✅ Created: requirements.txt")
        
        return True
    
    def create_sample_config(self):
        """Create sample configuration file"""
        print_section("Creating Configuration File")
        
        config = {
            "model_settings": {
                "sentiment_ensemble": True,
                "aspect_confidence_threshold": 0.3,
                "device": "auto"
            },
            "business_priorities": {
                "user_experience": 1.5,
                "performance": 1.3,
                "tracking_accuracy": 1.2,
                "delivery_issues": 1.1,
                "interface_design": 1.0,
                "general_satisfaction": 0.8
            },
            "data_settings": {
                "data_dir": "data",
                "test_results_dir": "test_results",
                "model_cache_dir": "models/cache"
            }
        }
        
        config_file = self.project_root / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Created: config.json")
        return True
    
    def print_setup_summary(self, results):
        """Print setup summary and next steps"""
        print_section("SETUP SUMMARY")
        
        all_passed = all(results.values())
        
        print("\n📋 Setup Results:")
        for step, passed in results.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {step}")
        
        if all_passed:
            print_colored("\n🎉 SETUP COMPLETED SUCCESSFULLY!", Colors.GREEN + Colors.BOLD)
            
            print_colored("\n📚 Next Steps:", Colors.CYAN)
            print("1. Test the models:")
            print("   python complete_model_pipeline_test.py")
            print("\n2. Run the Streamlit app:")
            print("   streamlit run app.py")
            print("\n3. Analyze FedEx data:")
            print("   python src/fedex_scraper.py")
            print("\n4. Check the documentation:")
            print("   See README.md for detailed usage")
            
            print_colored("\n🚀 Quick Test Commands:", Colors.CYAN)
            print("# Test sentiment analysis:")
            print('python -c "from src.models.enhanced_sentiment_classifier import EnhancedSentimentClassifier; model = EnhancedSentimentClassifier(); print(model.analyze_sentiment(\'Great product!\'))"')
            
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
        print_colored("\n✅ Your colleague can now use the repository!", Colors.GREEN + Colors.BOLD)
        print_colored("All models are set up and ready to use.", Colors.GREEN)
    else:
        print_colored("\n⚠️ Setup needs attention. See errors above.", Colors.WARNING)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
    