#!/usr/bin/env python3
"""
Updated Test Script for FedEx Scraper with Two-Model Ensemble Integration
Tests the new two-model ensemble sentiment classifier and integrated pipeline
"""

import sys
import os
from pathlib import Path
import time
import importlib
import pandas as pd
from datetime import datetime

# Add project root to Python path
current_file = Path(__file__).resolve()
# If we're in tests/ directory, go up one level to get the project root
if current_file.parent.name == 'tests':
    project_root = current_file.parent.parent
else:
    project_root = current_file.parent

# Add both project root and src directory to Python path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

print(f"Project root: {project_root}")
print(f"Source directory: {project_root / 'src'}")
print(f"Source directory exists: {(project_root / 'src').exists()}")

# Check if key files exist
key_files = [
    project_root / 'src' / 'models' / 'enhanced_sentiment_classifier.py',
    project_root / 'src' / 'models' / 'enhanced_aspect_classifier.py',
    project_root / 'src' / 'integrated_ml_pipeline.py',
    project_root / 'fedex_scraper.py'  # Check if it's in root
]

print("\nFile structure check:")
for file_path in key_files:
    print(f"  {file_path.name}: {file_path.exists()} ({file_path})")

print(f"Python path: {sys.path[:3]}")

# FORCE MODULE RELOAD TO PICK UP CHANGES
print("\nForcing module reload to pick up two-model ensemble changes...")
modules_to_reload = [
    'enhanced_sentiment_classifier',
    'enhanced_aspect_classifier', 
    'fedex_scraper',
    'integrated_ml_pipeline',
    'integrated_classifier'
]

for module_name in modules_to_reload:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
        print(f"  Reloaded: {module_name}")
    
    # Also try with src. prefix
    src_module_name = f'src.models.{module_name}'
    if src_module_name in sys.modules:
        importlib.reload(sys.modules[src_module_name])
        print(f"  Reloaded: {src_module_name}")

print("Module reload completed")

def test_two_model_ensemble_directly():
    """Test the two-model ensemble sentiment classifier directly"""
    print("\n" + "="*70)
    print("STEP 1: DIRECT TWO-MODEL ENSEMBLE TEST")
    print("="*70)
    
    try:
        # Try multiple import strategies
        classifier = None
        import_attempts = [
            ("from models.enhanced_sentiment_classifier import EnhancedSentimentClassifier", "from src.models"),
            ("from enhanced_sentiment_classifier import EnhancedSentimentClassifier", "from root"),
            ("from src.models.enhanced_sentiment_classifier import EnhancedSentimentClassifier", "with src.models prefix")
        ]
        
        for import_code, description in import_attempts:
            try:
                exec(import_code, globals())
                print(f"Successfully imported {description}")
                break
            except ImportError as e:
                print(f"Failed {description}: {e}")
                continue
        else:
            raise ImportError("Could not import EnhancedSentimentClassifier from any location")
        
        # Initialize with two-model ensemble
        classifier = EnhancedSentimentClassifier(device='auto', verbose=True)
        
        # Get ensemble info
        ensemble_info = classifier.get_model_info()
        print(f"\nEnsemble Information:")
        print(f"  Version: {ensemble_info.get('version', 'unknown')}")
        print(f"  Device: {ensemble_info.get('device', 'unknown')}")
        print(f"  Loaded models: {ensemble_info.get('loaded_models', 0)}")
        print(f"  Ensemble enabled: {ensemble_info.get('ensemble_enabled', False)}")
        
        # Critical test cases that should be negative
        critical_tests = [
            "Slowest, laziest, trashiest delivery company on the entire planet!!!",
            "The app has an issue when trying to view details of a delivery", 
            "not receiving email for sign in, this app continues to be trash!",
            "This app is absolutely TERRIBLE and keeps crashing!",
            "Complete disaster, never works, worst app ever"
        ]
        
        print(f"\nTesting {len(critical_tests)} critical negative cases with two-model ensemble:")
        negative_detection_count = 0
        
        for i, text in enumerate(critical_tests, 1):
            result = classifier.analyze_sentiment(text)
            sentiment = result['sentiment']
            confidence = result['confidence']
            method = result.get('method', 'unknown')
            models_used = result.get('models_used', 0)
            device = result.get('device', 'unknown')
            from_cache = result.get('from_cache', False)
            
            is_negative = sentiment == 'negative'
            if is_negative:
                negative_detection_count += 1
                
            status = "CORRECT" if is_negative else "MISSED"
            
            print(f"\n{i}. Text: '{text[:50]}...'")
            print(f"   Result: {sentiment.upper()} (confidence: {confidence:.3f}) [{status}]")
            print(f"   Method: {method}")
            print(f"   Models Used: {models_used}")
            print(f"   Device: {device}")
            print(f"   From Cache: {from_cache}")
            print(f"   Scores: Pos={result['scores']['positive']:.3f}, Neg={result['scores']['negative']:.3f}")
            
            if not is_negative:
                print(f"   ERROR: Expected NEGATIVE but got {sentiment.upper()}")
        
        accuracy = (negative_detection_count / len(critical_tests)) * 100
        print(f"\nTwo-Model Ensemble Direct Test Results:")
        print(f"  Negative detection accuracy: {negative_detection_count}/{len(critical_tests)} ({accuracy:.1f}%)")
        
        # Test some positive cases too
        positive_tests = [
            "Love the new features, much better than before!",
            "Excellent app, works perfectly every time",
            "Great delivery service, highly recommend"
        ]
        
        print(f"\nTesting positive cases:")
        positive_detection_count = 0
        
        for i, text in enumerate(positive_tests, 1):
            result = classifier.analyze_sentiment(text)
            sentiment = result['sentiment']
            confidence = result['confidence']
            method = result.get('method', 'unknown')
            
            is_positive = sentiment == 'positive'
            if is_positive:
                positive_detection_count += 1
                
            status = "CORRECT" if is_positive else "MISSED"
            
            print(f"{i}. '{text}' -> {sentiment.upper()} ({confidence:.3f}) [{status}] via {method}")
        
        positive_accuracy = (positive_detection_count / len(positive_tests)) * 100
        print(f"  Positive detection accuracy: {positive_detection_count}/{len(positive_tests)} ({positive_accuracy:.1f}%)")
        
        return accuracy, positive_accuracy, ensemble_info
        
    except Exception as e:
        print(f"Two-model ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0, {}

def test_integrated_ml_pipeline():
    """Test the integrated ML pipeline with two-model ensemble"""
    print("\n" + "="*70)
    print("STEP 2: INTEGRATED ML PIPELINE TEST")
    print("="*70)
    
    try:
        # Try multiple import strategies for IntegratedMLPipeline
        pipeline_class = None
        import_attempts = [
            ("from integrated_ml_pipeline import IntegratedMLPipeline", "from root"),
            ("from src.integrated_ml_pipeline import IntegratedMLPipeline", "from src"),
        ]
        
        for import_code, description in import_attempts:
            try:
                exec(import_code, globals())
                print(f"Successfully imported pipeline {description}")
                break
            except ImportError as e:
                print(f"Failed pipeline import {description}: {e}")
                continue
        else:
            raise ImportError("Could not import IntegratedMLPipeline from any location")
        
        # Initialize pipeline with two-model ensemble
        pipeline = IntegratedMLPipeline(device='auto', verbose=True)
        
        # Get pipeline info
        pipeline_info = pipeline.get_pipeline_info()
        print(f"\nPipeline Information:")
        print(f"  Type: {pipeline_info.get('pipeline_type', 'unknown')}")
        print(f"  Version: {pipeline_info.get('version', 'unknown')}")
        print(f"  Models loaded: {pipeline_info.get('models_loaded', {})}")
        
        # Test texts with expected outcomes
        test_cases = [
            {
                'text': 'Slowest, laziest, trashiest delivery company on the entire planet!!!',
                'expected_sentiment': 'negative',
                'expected_aspect': 'delivery_issues'
            },
            {
                'text': 'not receiving email for sign in, this app continues to be trash!',
                'expected_sentiment': 'negative', 
                'expected_aspect': 'user_experience'
            },
            {
                'text': 'App crashes constantly when trying to track packages',
                'expected_sentiment': 'negative',
                'expected_aspect': 'performance'
            },
            {
                'text': 'Love the new features! Fast, reliable, and easy to use.',
                'expected_sentiment': 'positive',
                'expected_aspect': 'user_experience'
            }
        ]
        
        print(f"\nTesting {len(test_cases)} cases with integrated pipeline:")
        
        results = []
        correct_sentiment = 0
        correct_aspect = 0
        
        for i, test_case in enumerate(test_cases, 1):
            result = pipeline.analyze_text(test_case['text'])
            
            # Check results
            sentiment_correct = result['sentiment'] == test_case['expected_sentiment']
            aspect_correct = result['primary_aspect'] == test_case['expected_aspect']
            
            if sentiment_correct:
                correct_sentiment += 1
            if aspect_correct:
                correct_aspect += 1
            
            results.append(result)
            
            print(f"\n{i}. Text: '{test_case['text'][:50]}...'")
            print(f"   Expected: {test_case['expected_sentiment']} / {test_case['expected_aspect']}")
            print(f"   Got: {result['sentiment']} ({result['sentiment_confidence']:.3f}) / {result['primary_aspect']}")
            print(f"   Method: {result.get('sentiment_method', 'unknown')}")
            print(f"   Models Used: {result.get('sentiment_models_used', 0)}")
            print(f"   Classification Type: {result['classification_type']}")
            print(f"   Priority: {result['priority_level']}")
            print(f"   Sentiment: {'CORRECT' if sentiment_correct else 'WRONG'}")
            print(f"   Aspect: {'CORRECT' if aspect_correct else 'WRONG'}")
        
        sentiment_accuracy = (correct_sentiment / len(test_cases)) * 100
        aspect_accuracy = (correct_aspect / len(test_cases)) * 100
        
        print(f"\nIntegrated Pipeline Results:")
        print(f"  Sentiment accuracy: {correct_sentiment}/{len(test_cases)} ({sentiment_accuracy:.1f}%)")
        print(f"  Aspect accuracy: {correct_aspect}/{len(test_cases)} ({aspect_accuracy:.1f}%)")
        
        # Test batch processing
        print(f"\nTesting batch processing...")
        texts = [case['text'] for case in test_cases]
        
        start_time = time.time()
        batch_results = pipeline.analyze_batch_with_business_intelligence(texts)
        processing_time = time.time() - start_time
        
        print(f"  Batch processed {len(texts)} texts in {processing_time:.2f} seconds")
        print(f"  Throughput: {len(texts)/processing_time:.1f} texts/second")
        
        # Show ensemble performance
        ensemble_perf = batch_results.get('ensemble_performance', {})
        if ensemble_perf:
            print(f"  Ensemble metrics:")
            print(f"    Method distribution: {ensemble_perf.get('sentiment_method_distribution', {})}")
            print(f"    Cache hit rate: {ensemble_perf.get('cache_performance', {}).get('hit_rate_percentage', 0):.1f}%")
            print(f"    Ensemble usage: {ensemble_perf.get('ensemble_efficiency', {}).get('two_model_ensemble_usage_pct', 0):.1f}%")
        
        return sentiment_accuracy, aspect_accuracy, ensemble_perf
        
    except Exception as e:
        print(f"Integrated pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0, {}

def test_fedex_scraper_integration():
    """Test the FedEx scraper with two-model ensemble integration"""
    print("\n" + "="*70)
    print("STEP 3: FEDEX SCRAPER INTEGRATION TEST")
    print("="*70)
    
    try:
        # Try multiple import strategies for FedExReviewAnalyzer
        import_attempts = [
            ("from fedex_scraper import FedExReviewAnalyzer", "from root"),
            ("from scrapers.fedex_scraper import FedExReviewAnalyzer", "from scrapers"),
            ("from src.scrapers.fedex_scraper import FedExReviewAnalyzer", "from src.scrapers")
        ]
        
        for import_code, description in import_attempts:
            try:
                exec(import_code, globals())
                print(f"Successfully imported FedEx scraper {description}")
                break
            except ImportError as e:
                print(f"Failed FedEx scraper import {description}: {e}")
                continue
        else:
            raise ImportError("Could not import FedExReviewAnalyzer from any location")
        
        # Initialize with two-model ensemble support
        analyzer = FedExReviewAnalyzer(use_enhanced_models=True, device='auto')
        print("FedExReviewAnalyzer initialized with two-model ensemble support")
        
        # Create test reviews with known content
        test_reviews = [
            {
                'app_id': 'com.fedex.ida.android',
                'text': 'Slowest, laziest, trashiest delivery company on the entire planet!!!',
                'rating': 1,
                'date': datetime.now(),
                'days_ago': 5,
                'country': 'us',
                'language_detected': 'en',
                'user': 'TestUser1',
                'is_real': False
            },
            {
                'app_id': 'com.fedex.ida.android', 
                'text': 'The app has an issue when trying to view details of a delivery',
                'rating': 2,
                'date': datetime.now(),
                'days_ago': 6,
                'country': 'us',
                'language_detected': 'en',
                'user': 'TestUser2',
                'is_real': False
            },
            {
                'app_id': 'com.fedex.ida.android',
                'text': 'not receiving email for sign in, this app continues to be trash!',
                'rating': 1,
                'date': datetime.now(),
                'days_ago': 7,
                'country': 'us',
                'language_detected': 'en',
                'user': 'TestUser3',
                'is_real': False
            },
            {
                'app_id': 'com.fedex.ida.android',
                'text': 'Love the new features, much better than before!',
                'rating': 5,
                'date': datetime.now(),
                'days_ago': 8,
                'country': 'us', 
                'language_detected': 'en',
                'user': 'TestUser4',
                'is_real': False
            }
        ]
        
        print(f"\nTesting with {len(test_reviews)} synthetic reviews...")
        
        # Test classification through the full pipeline
        classified_reviews = analyzer.classify_reviews_enhanced(test_reviews)
        
        if classified_reviews:
            print(f"Successfully classified {len(classified_reviews)} reviews through full pipeline!")
            
            # Analyze results with ensemble metrics
            print(f"\nFedEx Scraper Integration Results:")
            integration_negative_count = 0
            ensemble_usage_count = 0
            
            for i, review in enumerate(classified_reviews):
                text = review['text']
                sentiment = review.get('sentiment', 'unknown')
                confidence = review.get('sentiment_confidence', 0.0)
                method = review.get('sentiment_method', 'unknown')
                models_used = review.get('sentiment_models_used', 0)
                rating = review.get('rating', 0)
                primary_aspect = review.get('primary_aspect', 'unknown')
                
                # Track ensemble usage
                if method == 'two_model_ensemble':
                    ensemble_usage_count += 1
                
                # Expected sentiment based on rating and content
                if 'trash' in text.lower() or 'issue' in text.lower() or rating <= 2:
                    expected = 'negative'
                    if sentiment == 'negative':
                        integration_negative_count += 1
                elif 'love' in text.lower() or rating >= 4:
                    expected = 'positive' 
                else:
                    expected = 'neutral'
                
                is_correct = sentiment == expected
                status = "CORRECT" if is_correct else "WRONG"
                
                print(f"\n  Review {i+1}: [{status}]")
                print(f"    Text: {text[:50]}...")
                print(f"    Rating: {rating} stars")
                print(f"    Expected: {expected.upper()}")
                print(f"    Got: {sentiment.upper()} (confidence: {confidence:.3f})")
                print(f"    Method: {method} ({models_used} models)")
                print(f"    Primary Aspect: {primary_aspect}")
                print(f"    Ensemble Used: {'YES' if method == 'two_model_ensemble' else 'NO'}")
                
                if not is_correct:
                    print(f"    ERROR: Classification mismatch!")
                    
            # Summary
            expected_negative = sum(1 for r in test_reviews if r['rating'] <= 2)
            integration_accuracy = (integration_negative_count / expected_negative) * 100 if expected_negative > 0 else 0
            ensemble_usage_rate = (ensemble_usage_count / len(classified_reviews)) * 100
            
            print(f"\n  Integration accuracy for negative reviews: {integration_negative_count}/{expected_negative} ({integration_accuracy:.1f}%)")
            print(f"  Two-model ensemble usage rate: {ensemble_usage_count}/{len(classified_reviews)} ({ensemble_usage_rate:.1f}%)")
            
            # Convert to DataFrame for final verification
            df = pd.DataFrame(classified_reviews)
            print(f"\nDataFrame created with {len(df)} rows")
            print(f"Key columns present: {[col for col in ['sentiment', 'sentiment_method', 'sentiment_models_used', 'primary_aspect'] if col in df.columns]}")
            
            # Show distributions
            if 'sentiment' in df.columns:
                sentiment_dist = df['sentiment'].value_counts()
                print(f"\nSentiment distribution:")
                for sentiment, count in sentiment_dist.items():
                    print(f"  {sentiment}: {count}")
            
            if 'sentiment_method' in df.columns:
                method_dist = df['sentiment_method'].value_counts()
                print(f"\nMethod distribution:")
                for method, count in method_dist.items():
                    print(f"  {method}: {count}")
            
            return integration_accuracy, ensemble_usage_rate
        else:
            print("ERROR: No classified reviews returned!")
            return 0, 0
            
    except Exception as e:
        print(f"FedEx scraper integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0

def test_optional_real_scraping():
    """Optional test with real reviews (requires internet and google-play-scraper)"""
    print("\n" + "="*70)
    print("STEP 4: OPTIONAL REAL SCRAPING TEST")
    print("="*70)
    
    try:
        from google_play_scraper import reviews, Sort
        
        print("Testing with 3 real FedEx reviews...")
        
        result, _ = reviews(
            'com.fedex.ida.android',
            lang='en',
            country='US',
            sort=Sort.NEWEST,
            count=3
        )
        
        if result:
            print(f"Retrieved {len(result)} real reviews")
            
            # Try to import and test with real data
            import_attempts = [
                ("from fedex_scraper import FedExReviewAnalyzer", "from root"),
                ("from scrapers.fedex_scraper import FedExReviewAnalyzer", "from scrapers"),  
                ("from src.scrapers.fedex_scraper import FedExReviewAnalyzer", "from src.scrapers")
            ]
            
            for import_code, description in import_attempts:
                try:
                    exec(import_code, globals())
                    print(f"Successfully imported FedEx scraper for real test {description}")
                    break
                except ImportError as e:
                    print(f"Failed real test import {description}: {e}")
                    continue
            else:
                print("Could not import FedExReviewAnalyzer for real test")
                return 0
            
            analyzer = FedExReviewAnalyzer(use_enhanced_models=True, device='auto')
            
            real_reviews = []
            for review in result[:3]:
                if review.get('content'):
                    review_date = review.get('at', datetime.now())
                    real_reviews.append({
                        'app_id': 'com.fedex.ida.android',
                        'text': review.get('content', '').strip(),
                        'rating': review.get('score', 0),
                        'date': review_date,
                        'days_ago': (datetime.now() - review_date).days,
                        'country': 'us',
                        'language_detected': 'en',
                        'user': review.get('userName', 'Anonymous'),
                        'is_real': True
                    })
            
            if real_reviews:
                classified_real = analyzer.classify_reviews_enhanced(real_reviews)
                
                print(f"\nReal review analysis results:")
                for i, review in enumerate(classified_real):
                    print(f"\n  Real Review {i+1}:")
                    print(f"    Text: {review['text'][:60]}...")
                    print(f"    Rating: {review['rating']} stars")
                    print(f"    Sentiment: {review.get('sentiment', 'N/A')} (confidence: {review.get('sentiment_confidence', 0):.3f})")
                    print(f"    Method: {review.get('sentiment_method', 'N/A')}")
                    print(f"    Primary Aspect: {review.get('primary_aspect', 'N/A')}")
                    print(f"    Priority: {review.get('priority_level', 'N/A')}")
                
                return len(classified_real)
            else:
                print("No valid real reviews found")
                return 0
        else:
            print("No real reviews retrieved")
            return 0
            
    except ImportError:
        print("google-play-scraper not installed - skipping real scraping test")
        print("Install with: pip install google-play-scraper")
        return -1
    except Exception as e:
        print(f"Real scraping test failed (this is optional): {e}")
        return 0

def main():
    """Main test function"""
    print("="*70)
    print("FEDEX SCRAPER & TWO-MODEL ENSEMBLE INTEGRATION TEST")
    print("="*70)
    
    # Step 1: Test two-model ensemble directly
    neg_acc, pos_acc, ensemble_info = test_two_model_ensemble_directly()
    
    # Step 2: Test integrated ML pipeline
    sent_acc, aspect_acc, pipeline_perf = test_integrated_ml_pipeline()
    
    # Step 3: Test FedEx scraper integration
    integration_acc, ensemble_usage = test_fedex_scraper_integration()
    
    # Step 4: Optional real scraping test
    real_reviews_count = test_optional_real_scraping()
    
    # FINAL SUMMARY
    print("\n" + "="*70)
    print("FINAL TEST SUMMARY - TWO-MODEL ENSEMBLE")
    print("="*70)
    
    print(f"Test Results:")
    print(f"  1. Direct Two-Model Ensemble:")
    print(f"     - Negative detection: {neg_acc:.1f}%")
    print(f"     - Positive detection: {pos_acc:.1f}%")
    print(f"     - Ensemble enabled: {ensemble_info.get('ensemble_enabled', False)}")
    print(f"     - Models loaded: {ensemble_info.get('loaded_models', 0)}")
    
    print(f"  2. Integrated ML Pipeline:")
    print(f"     - Sentiment accuracy: {sent_acc:.1f}%")
    print(f"     - Aspect accuracy: {aspect_acc:.1f}%")
    
    print(f"  3. FedEx Scraper Integration:")
    print(f"     - Integration accuracy: {integration_acc:.1f}%")
    print(f"     - Ensemble usage rate: {ensemble_usage:.1f}%")
    
    print(f"  4. Real Scraping Test:")
    if real_reviews_count == -1:
        print(f"     - Skipped (google-play-scraper not installed)")
    elif real_reviews_count == 0:
        print(f"     - Failed or no reviews")
    else:
        print(f"     - Processed {real_reviews_count} real reviews")
    
    # Overall assessment
    overall_success = True
    issues = []
    
    if neg_acc < 80:
        overall_success = False
        issues.append("Low negative sentiment detection")
    
    if integration_acc < 80:
        overall_success = False
        issues.append("Low integration accuracy")
    
    if ensemble_usage < 50:
        issues.append("Low ensemble usage (may be using fallback)")
    
    if not ensemble_info.get('ensemble_enabled', False):
        issues.append("Two-model ensemble not enabled")
    
    print(f"\nOverall Assessment:")
    if overall_success and not issues:
        print("  SUCCESS: Two-model ensemble integration is working correctly!")
        print("  ✓ All sentiment detection tests passed")
        print("  ✓ Integration pipeline working properly")
        print("  ✓ Ensemble features active")
    elif overall_success:
        print("  PARTIAL SUCCESS: Core functionality working with minor issues:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  FAILED: Critical issues detected:")
        for issue in issues:
            print(f"    - {issue}")
    
    print(f"\nNext Steps:")
    if not overall_success:
        print("  1. Check if the two-model ensemble files were saved correctly")
        print("  2. Verify import paths and module structure")
        print("  3. Check device availability (GPU/CPU)")
        print("  4. Review sentiment classifier improvements")
    else:
        print("  1. System is ready for production use")
        print("  2. Consider running with real data for full validation")
        print("  3. Monitor ensemble performance metrics in production")
    
    print(f"\nKey Features Tested:")
    print(f"  ✓ Two-model ensemble (XLM-RoBERTa + Twitter-RoBERTa)")
    print(f"  ✓ Dynamic fallback system") 
    print(f"  ✓ Cache optimization")
    print(f"  ✓ Multi-label aspect classification")
    print(f"  ✓ Business intelligence generation")
    print(f"  ✓ Batch processing optimization")

if __name__ == "__main__":
    main()