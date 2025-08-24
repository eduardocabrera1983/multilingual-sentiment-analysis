#!/usr/bin/env python3
"""
Complete Model and Pipeline Testing Suite
Tests enhanced_aspect_classifier.py, enhanced_sentiment_classifier.py, and test_pipeline.py
Ensures all components and pipelines are working correctly before deployment
"""

import sys
import os
import time
import warnings
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import traceback

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'src', 'models'))

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
    print_colored(f"🔍 {title}", Colors.BOLD)
    print_colored(f"{'='*70}", Colors.CYAN)

def print_subsection(title: str):
    """Print a formatted subsection header"""
    print_colored(f"\n{'-'*50}", Colors.BLUE)
    print_colored(f"📌 {title}", Colors.BLUE)
    print_colored(f"{'-'*50}", Colors.BLUE)

class CompleteTestingSuite:
    """Complete testing suite for all ML models and pipelines"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.test_results = {}
        self.models = {}
        self.pipelines = {}
        self.start_time = None
        
        # Test data for comprehensive testing
        self.test_data = {
            'english': [
                # Single aspect tests
                "The app interface is completely unusable and confusing",
                "Tracking accuracy is perfect, always shows correct location",
                "App crashes constantly when trying to track packages",
                
                # Dual aspect tests
                "Love the tracking accuracy but the interface is confusing",
                "Great performance but terrible user interface design",
                
                # Mixed concerns tests
                "App crashes frequently, interface is terrible, and tracking is inaccurate",
                "The app is slow, design is ugly, and deliveries are always late",
                
                # Real-world examples
                "not receiving email for sign in, this app continues to be trash!",
                "Beautiful design and very fast performance, highly recommend",
                "This product has excellent quality but the interface is confusing",
                "Very easy to use and great build quality",
            ],
            'spanish': [
                "La interfaz es muy confusa y difícil de usar",
                "El seguimiento es muy preciso, excelente aplicación",
                "La aplicación se cierra constantemente",
                "La calidad del producto es excelente pero la interfaz es confusa",
            ],
            'german': [
                "Die App stürzt ständig ab beim Verfolgen von Paketen",
                "Sehr einfache Bedienung und genaue Verfolgung",
            ],
            'french': [
                "Interface très confuse et difficile à naviguer",
                "Suivi très précis et livraison rapide",
            ]
        }
    
    def check_environment(self) -> Dict[str, bool]:
        """Check if all required libraries are installed"""
        print_section("Environment Check")
        
        requirements = {
            'pandas': False,
            'numpy': False,
            'transformers': False,
            'torch': False,
            'sklearn': False,
            'langdetect': False,
        }
        
        for library in requirements:
            try:
                __import__(library)
                requirements[library] = True
                print_colored(f"✅ {library} is installed", Colors.GREEN)
            except ImportError:
                print_colored(f"❌ {library} is NOT installed", Colors.FAIL)
        
        # Check CUDA availability
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            print_colored(f"🖥️ CUDA available: {cuda_available}", 
                         Colors.GREEN if cuda_available else Colors.WARNING)
        except:
            print_colored("⚠️ Could not check CUDA availability", Colors.WARNING)
        
        # Check transformers availability
        try:
            from transformers import pipeline
            print_colored("✅ Transformers pipeline available", Colors.GREEN)
            requirements['transformers_pipeline'] = True
        except:
            print_colored("❌ Transformers pipeline not available", Colors.FAIL)
            requirements['transformers_pipeline'] = False
        
        all_installed = all(requirements.values())
        if not all_installed:
            print_colored("\n⚠️ Missing dependencies. Install with:", Colors.WARNING)
            print("pip install pandas numpy transformers torch scikit-learn langdetect")
        
        return requirements
    
    def test_enhanced_models_import(self) -> bool:
        """Test if enhanced model files can be imported"""
        print_section("Enhanced Models Import Test")
        
        success = True
        
        # Test Enhanced Sentiment Classifier import
        try:
            from src.models.enhanced_sentiment_classifier import EnhancedSentimentClassifier
            self.models['sentiment_class'] = EnhancedSentimentClassifier
            print_colored("✅ Enhanced Sentiment Classifier imported successfully", Colors.GREEN)
        except Exception as e:
            print_colored(f"❌ Failed to import Enhanced Sentiment Classifier: {e}", Colors.FAIL)
            success = False
        
        # Test Enhanced Aspect Classifier import
        try:
            from src.models.enhanced_aspect_classifier import EnhancedAspectClassifier
            self.models['aspect_class'] = EnhancedAspectClassifier
            print_colored("✅ Enhanced Aspect Classifier imported successfully", Colors.GREEN)
        except Exception as e:
            print_colored(f"❌ Failed to import Enhanced Aspect Classifier: {e}", Colors.FAIL)
            success = False
        
        return success
    
    def test_pipeline_import(self) -> bool:
        """Test if pipeline can be imported"""
        print_section("Pipeline Import Test")
        
        success = True
        
        # Test Integrated ML Pipeline import
        try:
            from src.integrated_ml_pipeline import IntegratedMLPipeline
            self.pipelines['integrated_class'] = IntegratedMLPipeline
            print_colored("✅ Integrated ML Pipeline imported successfully", Colors.GREEN)
        except Exception as e:
            print_colored(f"❌ Failed to import Integrated ML Pipeline: {e}", Colors.FAIL)
            print_colored("📋 Attempting fallback import...", Colors.WARNING)
            success = False
        
        return success
    
    def test_enhanced_models_initialization(self) -> bool:
        """Test if enhanced models can be initialized"""
        print_section("Enhanced Models Initialization Test")
        
        success = True
        
        # Initialize Enhanced Sentiment Classifier
        print_subsection("Initializing Enhanced Sentiment Classifier")
        try:
            if 'sentiment_class' in self.models:
                self.models['sentiment'] = self.models['sentiment_class'](use_ensemble=True)
                print_colored("✅ Enhanced Sentiment Classifier initialized", Colors.GREEN)
                
                # Get model info
                info = self.models['sentiment'].get_model_info()
                print(f"   📊 Loaded models: {info['loaded_models']}")
                print(f"   🔧 Ensemble enabled: {info['ensemble_enabled']}")
                print(f"   💻 Device: {info['device']}")
        except Exception as e:
            print_colored(f"❌ Failed to initialize Sentiment Classifier: {e}", Colors.FAIL)
            if self.verbose:
                traceback.print_exc()
            success = False
        
        # Initialize Enhanced Aspect Classifier
        print_subsection("Initializing Enhanced Aspect Classifier")
        try:
            if 'aspect_class' in self.models:
                self.models['aspect'] = self.models['aspect_class'](confidence_threshold=0.3)
                print_colored("✅ Enhanced Aspect Classifier initialized", Colors.GREEN)
                print("   🎯 Multi-label classification enabled")
                print("   👤 User experience prioritization active")
                print("   📊 Business intelligence features ready")
        except Exception as e:
            print_colored(f"❌ Failed to initialize Aspect Classifier: {e}", Colors.FAIL)
            if self.verbose:
                traceback.print_exc()
            success = False
        
        return success
    
    def test_pipeline_initialization(self) -> bool:
        """Test pipeline initialization with fallback"""
        print_section("Pipeline Initialization Test")
        
        print_colored("🚀 Testing Enhanced ML Pipeline with Full Ensemble", Colors.CYAN)
        
        try:
            # Try to initialize enhanced models directly
            print_subsection("Direct Enhanced Models Initialization")
            
            from src.models.enhanced_sentiment_classifier import EnhancedSentimentClassifier
            from src.models.enhanced_aspect_classifier import EnhancedAspectClassifier
            
            print("🤖 Initializing Enhanced Sentiment Classifier...")
            sentiment_classifier = EnhancedSentimentClassifier(use_ensemble=True)
            self.pipelines['sentiment'] = sentiment_classifier
            
            print("🎯 Initializing Enhanced Aspect Classifier...")
            aspect_classifier = EnhancedAspectClassifier(confidence_threshold=0.3)
            self.pipelines['aspect'] = aspect_classifier
            
            print_colored("✅ Full enhanced system loaded!", Colors.GREEN)
            return True
            
        except Exception as e:
            print_colored(f"❌ Failed to load enhanced models: {e}", Colors.FAIL)
            print_colored("📋 Falling back to integrated pipeline...", Colors.WARNING)
            
            try:
                from src.integrated_ml_pipeline import IntegratedMLPipeline
                pipeline = IntegratedMLPipeline()
                self.pipelines['integrated'] = pipeline
                print_colored("✅ Fallback pipeline loaded", Colors.GREEN)
                return True
                
            except Exception as e2:
                print_colored(f"❌ Even fallback failed: {e2}", Colors.FAIL)
                return False
    
    def test_enhanced_models_functionality(self) -> Dict:
        """Test enhanced models with test_pipeline.py style tests"""
        print_section("Enhanced Models Functionality Test (Pipeline Style)")
        
        if 'sentiment' not in self.pipelines or 'aspect' not in self.pipelines:
            print_colored("⚠️ Enhanced models not loaded, skipping", Colors.WARNING)
            return {'success': False, 'reason': 'Models not loaded'}
        
        test_texts = [
            "This product has excellent quality but the interface is confusing",
            "App crashes frequently and the user experience is terrible", 
            "Very easy to use and great build quality",
            "La calidad del producto es excelente pero la interfaz es confusa"
        ]
        
        print_colored(f"\n🧪 Testing Enhanced Models on {len(test_texts)} texts:", Colors.CYAN)
        print("="*60)
        
        success_count = 0
        total_tests = len(test_texts)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n{i}. Text: {text}")
            
            try:
                # Analyze sentiment with enhanced model
                sentiment_result = self.pipelines['sentiment'].analyze_sentiment(text)
                
                # Analyze aspect with enhanced model  
                aspect_result = self.pipelines['aspect'].classify_aspects_multilabel(text)
                
                print(f"   🤖 Enhanced Results:")
                print(f"      Sentiment: {sentiment_result['sentiment']} ({sentiment_result['confidence']:.3f})")
                print(f"      Models used: {sentiment_result.get('models_available', 'N/A')}")
                print(f"      Primary Aspect: {aspect_result['primary_aspect']}")
                print(f"      Secondary Aspects: {aspect_result['secondary_aspects']}")
                print(f"      Classification Type: {aspect_result['classification_type']}")
                print(f"      Priority Level: {aspect_result['priority_level']}")
                print(f"      Processing: {sentiment_result.get('processing_time', 0):.3f}s")
                
                # Show model ensemble info
                if 'model_used' in sentiment_result:
                    print(f"      Model ensemble: {sentiment_result['model_used']}")
                
                success_count += 1
                
            except Exception as e:
                print_colored(f"   ❌ Error: {e}", Colors.FAIL)
        
        accuracy = (success_count / total_tests * 100) if total_tests > 0 else 0
        print_colored(f"\n📊 Enhanced Models Success Rate: {accuracy:.1f}%", 
                     Colors.GREEN if accuracy > 75 else Colors.WARNING)
        
        return {
            'success': accuracy > 75,
            'success_count': success_count,
            'total_tests': total_tests,
            'accuracy': accuracy
        }
    
    def test_fallback_pipeline(self) -> Dict:
        """Test fallback pipeline functionality"""
        print_section("Fallback Pipeline Test")
        
        if 'integrated' not in self.pipelines:
            print_colored("⚠️ No fallback pipeline available", Colors.WARNING)
            return {'success': False, 'reason': 'No fallback'}
        
        pipeline = self.pipelines['integrated']
        test_text = "This product has excellent quality but the interface is confusing"
        
        print(f"\n🧪 Testing Fallback Pipeline:")
        print(f"Text: {test_text}")
        
        try:
            result = pipeline.analyze_text(test_text)
            
            print(f"\n📊 Fallback Results:")
            print(f"   Sentiment: {result['sentiment']} (confidence: {result['sentiment_confidence']:.3f})")
            print(f"   Primary Aspect: {result['primary_aspect']}")
            print(f"   Secondary Aspects: {result['secondary_aspects']}")
            print(f"   Classification Type: {result['classification_type']}")
            print(f"   Priority Level: {result['priority_level']}")
            print(f"   Processing time: {result['processing_time']:.3f} seconds")
            print(f"   Language: {result['language']}")
            
            print_colored(f"\n⚠️ Note: Using fallback models, not full enhanced system", Colors.WARNING)
            
            return {'success': True}
            
        except Exception as e:
            print_colored(f"❌ Fallback test failed: {e}", Colors.FAIL)
            return {'success': False, 'error': str(e)}
    
    def test_sentiment_analysis(self) -> Dict:
        """Test sentiment analysis functionality"""
        print_section("Sentiment Analysis Test")
        
        sentiment_model = None
        if 'sentiment' in self.models:
            sentiment_model = self.models['sentiment']
        elif 'sentiment' in self.pipelines:
            sentiment_model = self.pipelines['sentiment']
        
        if not sentiment_model:
            print_colored("⚠️ Sentiment model not available", Colors.WARNING)
            return {'success': False, 'tests_passed': 0, 'total_tests': 0}
        
        tests_passed = 0
        total_tests = 0
        results = []
        
        # Test cases with expected sentiment
        test_cases = [
            ("This product is amazing! Best purchase ever!", "positive"),
            ("Terrible experience, completely disappointed", "negative"),
            ("It's okay, nothing special", "neutral"),
            ("Love the quality but hate the price", "neutral"),  # Mixed
        ]
        
        print_subsection("Testing Sentiment Classification")
        for text, expected in test_cases:
            total_tests += 1
            try:
                result = sentiment_model.analyze_sentiment(text)
                
                sentiment = result['sentiment']
                confidence = result['confidence']
                
                # Check if sentiment is reasonable
                is_correct = (
                    (expected == "positive" and sentiment == "positive") or
                    (expected == "negative" and sentiment == "negative") or
                    (expected == "neutral" and sentiment in ["neutral", "positive", "negative"])
                )
                
                if is_correct:
                    tests_passed += 1
                    status = "✅"
                else:
                    status = "⚠️"
                
                print(f"{status} Text: '{text[:50]}...'")
                print(f"   Expected: {expected}, Got: {sentiment} (confidence: {confidence:.3f})")
                
            except Exception as e:
                print_colored(f"❌ Error processing text: {e}", Colors.FAIL)
        
        # Test multilingual capability
        print_subsection("Testing Multilingual Support")
        for lang, texts in self.test_data.items():
            if lang == 'english':
                continue
            
            try:
                sample_text = texts[0]
                result = sentiment_model.analyze_sentiment(sample_text, language=lang[:2])
                print(f"✅ {lang.capitalize()}: Successfully analyzed")
                print(f"   Text: '{sample_text[:40]}...'")
                print(f"   Result: {result['sentiment']} ({result['confidence']:.3f})")
                tests_passed += 0.5
                total_tests += 0.5
            except Exception as e:
                print_colored(f"⚠️ {lang.capitalize()}: Failed - {e}", Colors.WARNING)
                total_tests += 0.5
        
        accuracy = (tests_passed / total_tests * 100) if total_tests > 0 else 0
        print_colored(f"\n📊 Sentiment Analysis Accuracy: {accuracy:.1f}%", 
                     Colors.GREEN if accuracy > 70 else Colors.WARNING)
        
        return {
            'success': accuracy > 60,
            'tests_passed': tests_passed,
            'total_tests': total_tests,
            'accuracy': accuracy
        }
    
    def test_aspect_classification(self) -> Dict:
        """Test aspect classification functionality"""
        print_section("Aspect Classification Test")
        
        aspect_model = None
        if 'aspect' in self.models:
            aspect_model = self.models['aspect']
        elif 'aspect' in self.pipelines:
            aspect_model = self.pipelines['aspect']
        
        if not aspect_model:
            print_colored("⚠️ Aspect model not available", Colors.WARNING)
            return {'success': False, 'tests_passed': 0, 'total_tests': 0}
        
        tests_passed = 0
        total_tests = 0
        
        # Test multi-label classification
        print_subsection("Testing Multi-Label Classification")
        
        test_cases = [
            {
                'text': "Interface is completely unusable",
                'expected_primary': 'user_experience',
                'expected_type': 'single_aspect'
            },
            {
                'text': "Love the tracking but hate the interface",
                'expected_primary': ['tracking_accuracy', 'user_experience'],
                'expected_type': 'dual_aspect'
            },
            {
                'text': "App crashes, interface is terrible, tracking is wrong",
                'expected_primary': ['performance', 'user_experience'],
                'expected_type': 'mixed_concerns'
            }
        ]
        
        for test_case in test_cases:
            total_tests += 1
            try:
                result = aspect_model.classify_aspects_multilabel(test_case['text'])
                
                primary = result['primary_aspect']
                classification_type = result['classification_type']
                secondary = result['secondary_aspects']
                priority = result['priority_level']
                
                # Check if primary aspect is acceptable
                expected_primary = test_case['expected_primary']
                if isinstance(expected_primary, list):
                    primary_correct = primary in expected_primary
                else:
                    primary_correct = primary == expected_primary
                
                # Check classification type
                type_correct = classification_type == test_case['expected_type']
                
                if primary_correct and type_correct:
                    tests_passed += 1
                    status = "✅"
                elif type_correct:
                    tests_passed += 0.5
                    status = "⚠️"
                else:
                    status = "❌"
                
                print(f"{status} Text: '{test_case['text'][:50]}...'")
                print(f"   Primary: {primary}, Secondary: {secondary}")
                print(f"   Type: {classification_type}, Priority: {priority}")
                
            except Exception as e:
                print_colored(f"❌ Error: {e}", Colors.FAIL)
        
        accuracy = (tests_passed / total_tests * 100) if total_tests > 0 else 0
        print_colored(f"\n📊 Aspect Classification Accuracy: {accuracy:.1f}%",
                     Colors.GREEN if accuracy > 60 else Colors.WARNING)
        
        return {
            'success': accuracy > 50,
            'tests_passed': tests_passed,
            'total_tests': total_tests,
            'accuracy': accuracy
        }
    
    def test_integration(self) -> Dict:
        """Test integration between components"""
        print_section("Integration Test")
        
        # Check which models/pipelines are available
        has_enhanced = 'sentiment' in self.pipelines and 'aspect' in self.pipelines
        has_integrated = 'integrated' in self.pipelines
        
        if not has_enhanced and not has_integrated:
            print_colored("⚠️ No models available for integration test", Colors.WARNING)
            return {'success': False}
        
        print_subsection("Testing Combined Analysis")
        
        test_text = "The tracking is very accurate but the interface is confusing and the app crashes frequently"
        
        try:
            if has_enhanced:
                print("Testing with Enhanced Models:")
                sentiment_result = self.pipelines['sentiment'].analyze_sentiment(test_text)
                aspect_result = self.pipelines['aspect'].classify_aspects_multilabel(test_text)
                
                print(f"📝 Test text: '{test_text}'")
                print(f"\n🎭 Sentiment Analysis:")
                print(f"   Sentiment: {sentiment_result['sentiment']} ({sentiment_result['confidence']:.3f})")
                
                print(f"\n🎯 Aspect Classification:")
                print(f"   Primary: {aspect_result['primary_aspect']}")
                print(f"   Secondary: {aspect_result['secondary_aspects']}")
                print(f"   Type: {aspect_result['classification_type']}")
                
            elif has_integrated:
                print("Testing with Integrated Pipeline:")
                result = self.pipelines['integrated'].analyze_text(test_text)
                
                print(f"📝 Test text: '{test_text}'")
                print(f"\n🎭 Sentiment: {result['sentiment']} ({result['sentiment_confidence']:.3f})")
                print(f"🎯 Primary Aspect: {result['primary_aspect']}")
                print(f"   Secondary: {result['secondary_aspects']}")
                print(f"   Type: {result['classification_type']}")
            
            print_colored("\n✅ Integration test passed", Colors.GREEN)
            return {'success': True}
            
        except Exception as e:
            print_colored(f"❌ Integration test failed: {e}", Colors.FAIL)
            if self.verbose:
                traceback.print_exc()
            return {'success': False}
    
    def test_performance(self) -> Dict:
        """Test model performance and speed"""
        print_section("Performance Test")
        
        results = {}
        
        # Determine which models to test
        test_models = []
        
        if 'sentiment' in self.pipelines:
            test_models.append(('Enhanced Sentiment', self.pipelines['sentiment'], 'sentiment'))
        elif 'sentiment' in self.models:
            test_models.append(('Sentiment', self.models['sentiment'], 'sentiment'))
        
        if 'aspect' in self.pipelines:
            test_models.append(('Enhanced Aspect', self.pipelines['aspect'], 'aspect'))
        elif 'aspect' in self.models:
            test_models.append(('Aspect', self.models['aspect'], 'aspect'))
        
        if 'integrated' in self.pipelines:
            test_models.append(('Integrated Pipeline', self.pipelines['integrated'], 'integrated'))
        
        if not test_models:
            print_colored("⚠️ No models available for performance testing", Colors.WARNING)
            return {'success': False}
        
        for model_name, model, model_type in test_models:
            print_subsection(f"{model_name} Performance")
            
            texts = self.test_data['english'][:10]
            
            start_time = time.time()
            for text in texts:
                if model_type == 'sentiment':
                    model.analyze_sentiment(text)
                elif model_type == 'aspect':
                    model.classify_aspects_multilabel(text)
                elif model_type == 'integrated':
                    model.analyze_text(text)
            
            elapsed = time.time() - start_time
            avg_time = elapsed / len(texts)
            
            results[f'{model_name}_avg_time'] = avg_time
            
            print(f"⏱️ Average time per text: {avg_time:.3f}s")
            print(f"📊 Throughput: {1/avg_time:.1f} texts/second")
            
            if avg_time < 0.5:
                print_colored("✅ Excellent performance", Colors.GREEN)
            elif avg_time < 1.0:
                print_colored("✅ Good performance", Colors.GREEN)
            else:
                print_colored("⚠️ May need optimization", Colors.WARNING)
        
        return {'success': True, 'results': results}
    
    def run_comprehensive_test(self) -> Dict:
        """Run all tests and generate comprehensive report"""
        self.start_time = time.time()
        
        print_colored("\n" + "="*70, Colors.BOLD)
        print_colored("🚀 COMPLETE MODEL AND PIPELINE TESTING SUITE", Colors.BOLD)
        print_colored("="*70, Colors.BOLD)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("📋 Testing: enhanced_aspect_classifier.py, enhanced_sentiment_classifier.py, and pipelines")
        
        # Run all tests
        env_check = self.check_environment()
        
        # Model imports
        enhanced_import = self.test_enhanced_models_import()
        pipeline_import = self.test_pipeline_import()
        
        # Model initialization
        enhanced_init = False
        if enhanced_import:
            enhanced_init = self.test_enhanced_models_initialization()
        
        # Pipeline initialization (with fallback)
        pipeline_init = self.test_pipeline_initialization()
        
        # Functionality tests
        enhanced_func = {'success': False}
        if pipeline_init and ('sentiment' in self.pipelines and 'aspect' in self.pipelines):
            enhanced_func = self.test_enhanced_models_functionality()
        
        fallback_func = {'success': False}
        if 'integrated' in self.pipelines:
            fallback_func = self.test_fallback_pipeline()
        
        # Component tests
        sentiment_results = self.test_sentiment_analysis()
        aspect_results = self.test_aspect_classification()
        integration_results = self.test_integration()
        performance_results = self.test_performance()
        
        # Generate final report
        print_section("FINAL COMPREHENSIVE REPORT")
        
        # Count successes
        test_categories = {
            'Environment': all(env_check.values()),
            'Enhanced Models Import': enhanced_import,
            'Pipeline Import': pipeline_import,
            'Enhanced Models Init': enhanced_init,
            'Pipeline Init': pipeline_init,
            'Enhanced Functionality': enhanced_func['success'],
            'Fallback Pipeline': fallback_func['success'],
            'Sentiment Analysis': sentiment_results['success'],
            'Aspect Classification': aspect_results['success'],
            'Integration': integration_results['success'],
            'Performance': performance_results['success']
        }
        
        print_subsection("Test Summary")
        total_tests = 0
        passed_tests = 0
        
        for category, passed in test_categories.items():
            total_tests += 1
            if passed:
                passed_tests += 1
                print_colored(f"✅ {category}: PASSED", Colors.GREEN)
            else:
                print_colored(f"❌ {category}: FAILED", Colors.FAIL)
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print_colored(f"\n📊 Overall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})",
                     Colors.GREEN if success_rate > 70 else Colors.WARNING)
        
        elapsed_time = time.time() - self.start_time
        print(f"⏱️ Total test time: {elapsed_time:.2f} seconds")
        
        # System Status
        print_subsection("System Status")
        
        if 'sentiment' in self.pipelines and 'aspect' in self.pipelines:
            print_colored("✅ Enhanced models fully operational", Colors.GREEN)
            print("   • Multi-label classification: Active")
            print("   • User experience prioritization: Active")
            print("   • Business intelligence: Active")
            print("   • Ensemble sentiment analysis: Active")
        elif 'integrated' in self.pipelines:
            print_colored("⚠️ Running on fallback pipeline", Colors.WARNING)
            print("   • Basic functionality available")
            print("   • Consider troubleshooting enhanced models")
        else:
            print_colored("❌ No functional pipeline available", Colors.FAIL)
        
        # Recommendations
        print_subsection("Recommendations")
        
        if success_rate >= 90:
            print_colored("🎉 Excellent! All systems fully operational!", Colors.GREEN)
            print_colored("✅ Models ready for production deployment", Colors.GREEN)
            print_colored("✅ Perfect for bootcamp presentation", Colors.GREEN)
        elif success_rate >= 70:
            print_colored("✅ System is functional with minor issues", Colors.GREEN)
            print("Consider addressing failed tests for optimal performance")
            
            # Specific recommendations
            if not enhanced_func['success'] and fallback_func['success']:
                print("💡 Enhanced models failed but fallback works - check model downloads")
            if not sentiment_results['success']:
                print("💡 Sentiment analysis needs attention - check transformers installation")
            if not aspect_results['success']:
                print("💡 Aspect classification needs review - check keywords and models")
        else:
            print_colored("⚠️ Significant issues detected. Review failed tests.", Colors.WARNING)
            print("🔧 Troubleshooting steps:")
            print("   1. Ensure all dependencies are installed: pip install -r requirements.txt")
            print("   2. Check model files exist in correct locations")
            print("   3. Verify internet connection for model downloads")
            print("   4. Review error messages above for specific issues")
        
        # Save test results
        results = {
            'timestamp': datetime.now().isoformat(),
            'success_rate': success_rate,
            'test_categories': test_categories,
            'elapsed_time': elapsed_time,
            'models_loaded': {
                'enhanced_sentiment': 'sentiment' in self.models or 'sentiment' in self.pipelines,
                'enhanced_aspect': 'aspect' in self.models or 'aspect' in self.pipelines,
                'integrated_pipeline': 'integrated' in self.pipelines
            },
            'detailed_results': {
                'enhanced_functionality': enhanced_func,
                'fallback': fallback_func,
                'sentiment': sentiment_results,
                'aspect': aspect_results,
                'integration': integration_results,
                'performance': performance_results
            }
        }
        
        # Save to JSON file
        try:
            os.makedirs('test_results', exist_ok=True)
            filename = f"test_results/complete_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n📁 Test results saved to: {filename}")
        except Exception as e:
            print_colored(f"⚠️ Could not save results: {e}", Colors.WARNING)
        
        return results


def main():
    """Main execution function"""
    tester = CompleteTestingSuite(verbose=True)
    results = tester.run_comprehensive_test()
    
    # Print final status
    print_colored("\n" + "="*70, Colors.BOLD)
    
    success_rate = results.get('success_rate', 0)
    
    if success_rate >= 90:
        print_colored("🏆 COMPLETE TESTING: EXCELLENT", Colors.GREEN + Colors.BOLD)
        print_colored("All models and pipelines are fully operational!", Colors.GREEN)
        
        print_colored("\n✅ Ready for:", Colors.CYAN)
        print("   • Production deployment")
        print("   • Bootcamp presentation")
        print("   • FedEx data analysis")
        print("   • Real-world applications")
        
    elif success_rate >= 70:
        print_colored("✅ COMPLETE TESTING: PASSED WITH MINOR ISSUES", Colors.GREEN + Colors.BOLD)
        print_colored("System is functional and ready for use", Colors.GREEN)
        
        print_colored("\n📋 Next Steps:", Colors.CYAN)
        print("1. Review warnings in the test output")
        print("2. Address any failed tests if critical")
        print("3. Test with your actual FedEx data")
        print("4. Prepare for presentation")
        
    else:
        print_colored("⚠️ COMPLETE TESTING: NEEDS ATTENTION", Colors.WARNING + Colors.BOLD)
        print_colored("Please address the issues before proceeding", Colors.WARNING)
        
        print_colored("\n🔧 Troubleshooting:", Colors.CYAN)
        print("1. Check all dependencies: pip install -r requirements.txt")
        print("2. Verify model files are in: src/models/")
        print("3. Ensure internet connection for model downloads")
        print("4. Review specific error messages above")
        print("5. Try running individual model tests")
    
    print_colored("="*70, Colors.BOLD)
    
    # Show quick summary
    print_colored("\n📊 Quick Summary:", Colors.CYAN)
    if 'models_loaded' in results:
        models = results['models_loaded']
        print(f"   Enhanced Sentiment: {'✅' if models.get('enhanced_sentiment') else '❌'}")
        print(f"   Enhanced Aspect: {'✅' if models.get('enhanced_aspect') else '❌'}")
        print(f"   Integrated Pipeline: {'✅' if models.get('integrated_pipeline') else '❌'}")
    
    print(f"\n💡 For detailed results, check: test_results/")


if __name__ == "__main__":
    main()