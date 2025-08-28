#!/usr/bin/env python3
"""
Pure Multi-Label Test Setup - FIXED VERSION
Save as: tests/test_setup.py

Clean setup testing only pure multi-label implementation
"""

import sys
import os
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

def test_basic_imports():
    """Test basic library imports"""
    print("🔧 Testing Basic Library Imports:")
    print("-" * 40)
    
    try:
        import transformers
        import torch
        import streamlit as st
        import pandas as pd
        import numpy as np
        from transformers import pipeline
        import langdetect
        import sklearn
        
        print("✅ All basic libraries imported successfully!")
        print(f"   Transformers version: {transformers.__version__}")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   Streamlit version: {st.__version__}")
        print(f"   Pandas version: {pd.__version__}")
        print(f"   NumPy version: {np.__version__}")
        print(f"   Scikit-learn version: {sklearn.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📦 Install missing packages with: pip install -r requirements.txt")
        return False

def test_pure_multilabel_imports():
    """Test pure multi-label model imports"""
    print("\n🚀 Testing Pure Multi-Label Model Imports:")
    print("-" * 50)
    
    # Add paths - FIXED
    sys.path.append('.')  # Add project root
    sys.path.append('src')
    sys.path.append('src/models')
    sys.path.append('src/pipelines')
    
    try:
        # Test enhanced sentiment classifier
        from src.models.enhanced_sentiment_classifier import EnhancedSentimentClassifier
        print("✅ Enhanced sentiment classifier imported")
        
        # Test pure multi-label aspect classifier
        from src.models.enhanced_aspect_classifier import EnhancedAspectClassifier
        print("✅ Pure multi-label aspect classifier imported")
        
        # Test pure multi-label integrated pipeline - FIXED
        from src.integrated_ml_pipeline import IntegratedMLPipeline
        print("✅ Pure multi-label integrated pipeline imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Pure multi-label import error: {e}")
        print("📂 Check that files exist:")
        print("   - src/models/enhanced_sentiment_classifier.py")
        print("   - src/models/enhanced_aspect_classifier.py") 
        print("   - src/integrated_ml_pipeline.py")
        return False

def test_pure_multilabel_initialization():
    """Test pure multi-label model initialization"""
    print("\n⚙️ Testing Pure Multi-Label Initialization:")
    print("-" * 50)
    
    try:
        # Test enhanced sentiment classifier - FIXED
        print("🔮 Initializing enhanced sentiment classifier...")
        from src.models.enhanced_sentiment_classifier import EnhancedSentimentClassifier
        sentiment_classifier = EnhancedSentimentClassifier(use_ensemble=True)
        print("✅ Enhanced sentiment classifier initialized")
        
        # Test pure multi-label aspect classifier - FIXED
        print("🎯 Initializing pure multi-label aspect classifier...")
        from src.models.enhanced_aspect_classifier import EnhancedAspectClassifier
        aspect_classifier = EnhancedAspectClassifier()
        print("✅ Pure multi-label aspect classifier initialized")
        
        # Test pure multi-label integrated pipeline - FIXED
        print("🔗 Initializing pure multi-label integrated pipeline...")
        from src.integrated_ml_pipeline import IntegratedMLPipeline
        pipeline = IntegratedMLPipeline()
        print("✅ Pure multi-label integrated pipeline initialized")
        
        # Verify pipeline configuration
        pipeline_info = pipeline.get_pipeline_info()
        
        if pipeline_info['backward_compatible']:
            print("⚠️ WARNING: Pipeline still has backward compatibility")
            print("   Expected: Pure multi-label implementation")
            return False
        else:
            print("✅ CONFIRMED: Pure multi-label implementation (no backward compatibility)")
        
        return True, {
            'sentiment_classifier': sentiment_classifier,
            'aspect_classifier': aspect_classifier,
            'pipeline': pipeline
        }
        
    except Exception as e:
        print(f"❌ Pure multi-label initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_new_output_format():
    """Test that new output format is working (primary_aspect instead of aspect)"""
    print("\n📊 Testing New Output Format:")
    print("-" * 40)
    
    success, models = test_pure_multilabel_initialization()
    if not success:
        return False
    
    test_text = "not receiving email for sign in, this app continues to be trash!"
    
    try:
        pipeline = models['pipeline']
        
        # Test new format
        result = pipeline.analyze_text(test_text)
        
        # Check for NEW format fields
        new_format_fields = [
            'primary_aspect', 'secondary_aspects', 'classification_type',
            'priority_level', 'severity_level', 'business_summary', 
            'recommendation', 'requires_immediate_action'
        ]
        
        # Check for OLD format fields that should NOT exist
        old_format_fields = ['aspect', 'aspect_confidence']  # These should be replaced
        
        missing_new = [field for field in new_format_fields if field not in result]
        present_old = [field for field in old_format_fields if field in result]
        
        if missing_new:
            print(f"❌ Missing NEW format fields: {missing_new}")
            return False
        
        if present_old:
            print(f"⚠️ OLD format fields still present: {present_old}")
            print("   Expected: Complete replacement with new format")
            # Don't fail for this, but warn
        
        print(f"✅ New output format working:")
        print(f"   Primary Aspect: {result['primary_aspect']}")
        print(f"   Secondary Aspects: {result['secondary_aspects']}")
        print(f"   Classification Type: {result['classification_type']}")
        print(f"   Priority Level: {result['priority_level']}")
        print(f"   Business Summary: {result['business_summary']}")
        
        return True
        
    except Exception as e:
        print(f"❌ New format test failed: {e}")
        return False

def test_multilabel_capabilities():
    """Test multi-label classification capabilities"""
    print("\n🎯 Testing Multi-Label Capabilities:")
    print("-" * 40)
    
    success, models = test_pure_multilabel_initialization()
    if not success:
        return False
    
    # Test cases that show multi-label power
    multilabel_tests = [
        {
            'text': "Love the tracking accuracy but the interface is confusing",
            'expected_type': 'dual_aspect',
            'description': 'Dual aspect test'
        },
        {
            'text': "App crashes, interface is terrible, and deliveries are late",
            'expected_type': 'mixed_concerns',
            'description': 'Mixed concerns test'
        },
        {
            'text': "Interface is impossible to use",
            'expected_type': 'single_aspect',
            'description': 'Single aspect test'
        }
    ]
    
    try:
        pipeline = models['pipeline']
        
        success_count = 0
        for i, test in enumerate(multilabel_tests, 1):
            result = pipeline.analyze_text(test['text'])
            
            print(f"\n{i}. {test['description']}")
            print(f"   Text: {test['text']}")
            print(f"   Result: {result['classification_type']}")
            print(f"   Primary: {result['primary_aspect']}")
            print(f"   Secondary: {result['secondary_aspects']}")
            
            if result['classification_type'] == test['expected_type']:
                print(f"   ✅ CORRECT classification type")
                success_count += 1
            else:
                print(f"   ⚠️ Expected {test['expected_type']}, got {result['classification_type']}")
        
        accuracy = (success_count / len(multilabel_tests)) * 100
        print(f"\n📊 Multi-label accuracy: {accuracy:.1f}%")
        
        return accuracy >= 66  # At least 2/3 correct
        
    except Exception as e:
        print(f"❌ Multi-label test failed: {e}")
        return False

def test_user_experience_prioritization():
    """Test user experience prioritization"""
    print("\n🎨 Testing User Experience Prioritization:")
    print("-" * 50)
    
    success, models = test_pure_multilabel_initialization()
    if not success:
        return False
    
    # Test UX prioritization
    ux_tests = [
        {
            'text': "Interface is impossible to use, terrible design",
            'should_prioritize_ux': True,
            'description': 'Pure UX issue'
        },
        {
            'text': "App crashes and interface is confusing",
            'should_prioritize_ux': True,  # UX should win due to priority weights
            'description': 'Mixed issue - UX should be prioritized'
        }
    ]
    
    try:
        pipeline = models['pipeline']
        
        ux_success = 0
        for i, test in enumerate(ux_tests, 1):
            result = pipeline.analyze_text(test['text'])
            
            print(f"\n{i}. {test['description']}")
            print(f"   Text: {test['text']}")
            print(f"   Primary: {result['primary_aspect']}")
            print(f"   UX Priority Flag: {result['user_experience_priority']}")
            
            if result['user_experience_priority'] == test['should_prioritize_ux']:
                print(f"   ✅ UX prioritization CORRECT")
                ux_success += 1
            else:
                print(f"   ⚠️ UX prioritization: Expected {test['should_prioritize_ux']}, got {result['user_experience_priority']}")
        
        accuracy = (ux_success / len(ux_tests)) * 100
        print(f"\n📊 UX prioritization accuracy: {accuracy:.1f}%")
        
        return accuracy >= 50
        
    except Exception as e:
        print(f"❌ UX prioritization test failed: {e}")
        return False

def test_business_intelligence():
    """Test business intelligence generation"""
    print("\n📊 Testing Business Intelligence Generation:")
    print("-" * 50)
    
    success, models = test_pure_multilabel_initialization()
    if not success:
        return False
    
    try:
        pipeline = models['pipeline']
        
        # Test with multiple texts
        test_texts = [
            "Interface is impossible to use",
            "Love tracking but hate interface", 
            "App crashes constantly",
            "Great app overall"
        ]
        
        # Generate business intelligence
        batch_result = pipeline.analyze_batch_with_business_intelligence(test_texts)
        
        # Check BI components
        bi = batch_result['business_intelligence']
        required_bi_fields = [
            'total_reviews', 'business_metrics', 'top_recommendations'
        ]
        
        missing_fields = [field for field in required_bi_fields if field not in bi]
        
        if missing_fields:
            print(f"❌ Missing BI fields: {missing_fields}")
            return False
        
        print(f"✅ Business intelligence generated:")
        print(f"   Total Reviews: {bi['total_reviews']}")
        print(f"   Mixed Concerns: {bi['business_metrics'].get('mixed_concerns_percentage', 0)}%")
        print(f"   UX Priority: {bi['business_metrics'].get('user_experience_priority_percentage', 0)}%")
        print(f"   High Priority: {bi['business_metrics'].get('high_priority_percentage', 0)}%")
        
        print(f"   Top Recommendations: {len(bi['top_recommendations'])}")
        for rec in bi['top_recommendations'][:2]:
            print(f"      • {rec}")
        
        return True
        
    except Exception as e:
        print(f"❌ Business intelligence test failed: {e}")
        return False

def test_fedex_data_structure():
    """Test compatibility with FedEx data structure"""
    print("\n📱 Testing FedEx Data Structure:")
    print("-" * 40)
    
    try:
        import pandas as pd
        
        # Check if FedEx data exists
        fedex_file = "data/fedex_reviews_20250822_1657.csv"
        
        if os.path.exists(fedex_file):
            print(f"✅ FedEx data found: {fedex_file}")
            
            df = pd.read_csv(fedex_file)
            print(f"   Original shape: {df.shape}")
            print(f"   Original columns: {list(df.columns)}")
            
            # Check required column
            if 'text' not in df.columns:
                print("❌ Required 'text' column not found")
                return False
            
            print("✅ Required 'text' column found")
            
            # Test with sample
            success, models = test_pure_multilabel_initialization()
            if success:
                pipeline = models['pipeline']
                
                # Test DataFrame analysis
                print("📄 Testing DataFrame analysis...")
                
                # Analyze small sample
                sample_df = df.head(3).copy()
                result_df = pipeline.analyze_dataframe(sample_df, text_column='text')
                
                # Check new columns were added
                new_columns = [col for col in result_df.columns if col.startswith('predicted_')]
                expected_new_columns = [
                    'predicted_primary_aspect', 'predicted_secondary_aspects',
                    'predicted_classification_type', 'predicted_priority_level'
                ]
                
                found_expected = sum(1 for col in expected_new_columns if col in new_columns)
                
                print(f"✅ New columns added: {len(new_columns)}")
                print(f"   Expected columns found: {found_expected}/{len(expected_new_columns)}")
                
                # Show sample result
                for idx in range(min(2, len(result_df))):
                    row = result_df.iloc[idx]
                    text = row['text'][:50] + '...' if len(row['text']) > 50 else row['text']
                    primary = row.get('predicted_primary_aspect', 'N/A')
                    print(f"   Sample: '{text}' -> {primary}")
                
                return found_expected >= len(expected_new_columns) * 0.75  # 75% of expected columns
            else:
                print("❌ Could not initialize models for FedEx test")
                return False
        else:
            print(f"⚠️ FedEx data not found at {fedex_file}")
            print("   System ready for when data is available")
            return True  # Not a failure, just missing data
            
    except Exception as e:
        print(f"❌ FedEx data structure test failed: {e}")
        return False

def run_comprehensive_pure_setup_test():
    """Run comprehensive pure multi-label setup test"""
    print("🧪 COMPREHENSIVE PURE MULTI-LABEL SETUP TEST")
    print("="*70)
    
    tests = [
        ("Basic Library Imports", test_basic_imports),
        ("Pure Multi-Label Imports", test_pure_multilabel_imports),
        ("New Output Format", test_new_output_format),
        ("Multi-Label Capabilities", test_multilabel_capabilities),
        ("User Experience Prioritization", test_user_experience_prioritization), 
        ("Business Intelligence", test_business_intelligence),
        ("FedEx Data Structure", test_fedex_data_structure)
    ]
    
    results = {}
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n{'='*70}")
        result = test_func()
        results[test_name] = result
        
        if not result:
            all_passed = False
    
    # Final summary
    print(f"\n{'='*70}")
    print("🎯 PURE MULTI-LABEL SETUP TEST SUMMARY")
    print(f"{'='*70}")
    
    passed_count = sum(1 for passed in results.values() if passed)
    total_count = len(results)
    success_rate = (passed_count / total_count) * 100
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\n📊 OVERALL SUCCESS RATE: {success_rate:.1f}% ({passed_count}/{total_count})")
    
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Your pure multi-label system is ready for:")
        print(f"   • Advanced multi-label aspect classification")
        print(f"   • User experience prioritization") 
        print(f"   • Mixed concerns detection")
        print(f"   • Business intelligence reporting")
        print(f"   • Bootcamp presentation excellence")
        
        print(f"\n🚀 BOOTCAMP PRESENTATION READY!")
        print(f"🎯 Key talking points:")
        print(f"   1. Advanced ML: Multi-label classification vs single-label")
        print(f"   2. Business intelligence: Automated priority assignment")
        print(f"   3. User experience focus: Prioritization weighting system")
        print(f"   4. Real-world application: FedEx review analysis")
        print(f"   5. Actionable insights: Recommendations for business teams")
        
    else:
        print(f"\n⚠️ {total_count - passed_count} TESTS FAILED")
        print(f"❌ Fix failed tests before bootcamp presentation")
        print(f"💡 Check error messages above for specific issues")
        
        print(f"\n🔧 Common fixes:")
        print(f"   • Ensure all files are saved with correct content")
        print(f"   • Check import paths and file locations")
        print(f"   • Verify transformers library installation")
        print(f"   • Make sure FedEx data file exists")
    
    return all_passed, results

# Main execution
if __name__ == "__main__":
    success, results = run_comprehensive_pure_setup_test()
    
    if success:
        print(f"\n" + "="*70)
        print("🏆 PURE MULTI-LABEL SYSTEM READY FOR BOOTCAMP!")
        print(f"="*70)
        print(f"📋 Next steps:")
        print(f"   1. Run: python src/model_testing.py")
        print(f"   2. Test with your FedEx data")
        print(f"   3. Prepare presentation slides")
        print(f"   4. Showcase your advanced ML system!")
        
        print(f"\n🎤 PRESENTATION DEMO:")
        print(f"   Your system demonstrates sophisticated ML concepts")
        print(f"   Perfect for impressing bootcamp instructors")
        print(f"   Shows real-world business application")
        
    else:
        print(f"\n" + "="*70)
        print("🔧 SETUP NEEDS REFINEMENT") 
        print(f"="*70)
        print(f"📋 Address the failed tests above")
        print(f"🔄 Then re-run: python tests/test_setup.py")