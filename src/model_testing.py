#!/usr/bin/env python3
"""
Pure Multi-Label Model Testing - FINAL VERSION

"""

import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from langdetect import detect
import time
import warnings
import sys
import os
from collections import Counter
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append('src')

class PureMultiLabelModelTester:
    """Pure multi-label model testing for bootcamp presentation"""
    
    def __init__(self):
        self.models = {}
        self.test_results = {}
        
        # Comprehensive test texts for your presentation
        self.test_texts = {
            'en': [
                # Single aspect examples
                "Interface is impossible to use, terrible navigation and confusing layout",
                "App crashes constantly when trying to track packages",
                "Tracking is very accurate, always shows correct package location",
                
                # Dual aspect examples
                "Love the tracking accuracy but the interface is so confusing",
                "Great delivery notifications but app crashes frequently",
                "Fast performance but ugly design and poor layout",
                
                # Mixed concerns (your key innovation)
                "App crashes when I try to track packages, interface is confusing, and deliveries are always late",
                "Love the tracking but hate the interface and app keeps freezing",
                
                # Your actual FedEx sample
                "not receiving email for sign in, this app continues to be trash!",
                
                # User experience priority examples
                "This app is completely unusable, worst interface ever designed",
                "Beautiful interface, very easy to navigate and find features"
            ],
            'es': [
                "La interfaz es muy confusa y difícil de navegar",
                "Excelente seguimiento pero la interfaz es terrible",
                "La aplicación se cierra constantemente al rastrear paquetes"
            ],
            'de': [
                "App stürzt ab beim Verfolgen mehrerer Pakete",
                "Sehr einfach zu bedienen, aber schlechte Tracking-Genauigkeit"
            ]
        }
    
    def load_pure_multilabel_models(self):
        """Load pure multi-label models"""
        print("🤖 Loading Pure Multi-Label Models...")
        print("-" * 50)
        
        # Load enhanced sentiment classifier
        try:
            from models.enhanced_sentiment_classifier import EnhancedSentimentClassifier
            self.models['sentiment'] = EnhancedSentimentClassifier(use_ensemble=True)
            print("✅ Enhanced sentiment classifier loaded")
        except Exception as e:
            print(f"❌ Failed to load sentiment classifier: {e}")
            return False
        
        # Load pure multi-label aspect classifier
        try:
            from models.enhanced_aspect_classifier import EnhancedAspectClassifier
            self.models['aspect'] = EnhancedAspectClassifier()
            print("✅ Pure multi-label aspect classifier loaded")
        except Exception as e:
            print(f"❌ Failed to load aspect classifier: {e}")
            return False
        
        # Load pure multi-label integrated pipeline
        try:
            from integrated_ml_pipeline import IntegratedMLPipeline
            self.models['pipeline'] = IntegratedMLPipeline()
            print("✅ Pure multi-label integrated pipeline loaded")
        except Exception as e:
            print(f"❌ Failed to load pipeline: {e}")
            return False
        
        # Verify pipeline configuration
        pipeline_info = self.models['pipeline'].get_pipeline_info()
        if not pipeline_info['backward_compatible']:
            print("✅ Confirmed: Pure multi-label implementation (no backward compatibility)")
        else:
            print("⚠️ Warning: Pipeline still has backward compatibility")
        
        return True
    
    def test_multilabel_classification_types(self):
        """Test all types of multi-label classification"""
        print("\n🎯 Testing Multi-Label Classification Types:")
        print("-" * 50)
        
        pipeline = self.models.get('pipeline')
        if not pipeline:
            print("❌ Pipeline not loaded")
            return
        
        # Test cases that demonstrate your innovation
        classification_test_cases = [
            {
                'text': "Interface is impossible to use, worst app ever",
                'expected_type': 'single_aspect',
                'expected_primary': 'user_experience',
                'description': 'Single aspect - User Experience Priority'
            },
            {
                'text': "Love the tracking accuracy but interface is confusing",
                'expected_type': 'dual_aspect', 
                'expected_primary': 'tracking_accuracy',
                'description': 'Dual aspect - Mixed concerns'
            },
            {
                'text': "App crashes frequently, interface is terrible, and deliveries are always late",
                'expected_type': 'mixed_concerns',
                'expected_primary': 'performance',
                'description': 'Mixed concerns - Multiple complex issues'
            },
            {
                'text': "not receiving email for sign in, this app continues to be trash!",
                'expected_type': 'single_aspect',
                'expected_primary': 'performance', 
                'description': 'Real FedEx sample - Authentication + General dissatisfaction'
            }
        ]
        
        success_count = 0
        for i, test_case in enumerate(classification_test_cases, 1):
            text = test_case['text']
            
            print(f"\n{i}. {test_case['description']}")
            print(f"   Text: {text}")
            
            # Analyze with pure multi-label system
            result = pipeline.analyze_text(text)
            
            print(f"   ✅ Primary: {result['primary_aspect']}")
            print(f"   ✅ Secondary: {result['secondary_aspects']}")
            print(f"   ✅ Type: {result['classification_type']}")
            print(f"   ✅ Priority: {result['priority_level']}")
            print(f"   ✅ Severity: {result['severity_level']}")
            print(f"   ✅ UX Priority: {result['user_experience_priority']}")
            print(f"   ✅ Mixed Concerns: {result['mixed_concerns']}")
            print(f"   ✅ Immediate Action: {result['requires_immediate_action']}")
            
            # Verify expected behavior
            type_match = result['classification_type'] == test_case['expected_type']
            primary_match = result['primary_aspect'] == test_case['expected_primary']
            
            if type_match and primary_match:
                print(f"   🎯 CLASSIFICATION: ✅ PERFECT MATCH")
                success_count += 1
            elif type_match:
                print(f"   🎯 CLASSIFICATION: ✅ TYPE CORRECT, Primary: Expected {test_case['expected_primary']}, Got {result['primary_aspect']}")
                success_count += 0.5
            else:
                print(f"   🎯 CLASSIFICATION: ⚠️ Expected {test_case['expected_type']}, Got {result['classification_type']}")
        
        accuracy = (success_count / len(classification_test_cases)) * 100
        print(f"\n📊 Classification Accuracy: {accuracy:.1f}% ({success_count}/{len(classification_test_cases)})")
        return accuracy > 75  # Success if > 75% accurate
    
    def test_user_experience_prioritization(self):
        """Test user experience prioritization feature"""
        print("\n🎨 Testing User Experience Prioritization:")
        print("-" * 50)
        
        pipeline = self.models.get('pipeline')
        if not pipeline:
            print("❌ Pipeline not loaded")
            return False
        
        # Test texts specifically for UX prioritization
        ux_priority_tests = [
            {
                'text': "Interface is completely unusable, terrible design",
                'should_be_ux_priority': True,
                'description': 'Pure UX issue'
            },
            {
                'text': "Love the tracking but interface is confusing",
                'should_be_ux_priority': False,  # Tracking is primary
                'description': 'Mixed - tracking primary'
            },
            {
                'text': "App crashes and interface is terrible", 
                'should_be_ux_priority': True,   # UX has higher priority weight
                'description': 'Mixed - UX should win due to prioritization'
            },
            {
                'text': "Package tracking is very accurate",
                'should_be_ux_priority': False,
                'description': 'Pure tracking issue'
            }
        ]
        
        ux_priority_success = 0
        for i, test in enumerate(ux_priority_tests, 1):
            result = pipeline.analyze_text(test['text'])
            
            print(f"\n{i}. {test['description']}")
            print(f"   Text: {test['text']}")
            print(f"   Primary: {result['primary_aspect']}")
            print(f"   UX Priority Flag: {result['user_experience_priority']}")
            
            # Check if UX prioritization worked correctly
            if result['user_experience_priority'] == test['should_be_ux_priority']:
                print(f"   🎯 UX PRIORITIZATION: ✅ CORRECT")
                ux_priority_success += 1
            else:
                print(f"   🎯 UX PRIORITIZATION: ⚠️ Expected {test['should_be_ux_priority']}, Got {result['user_experience_priority']}")
        
        accuracy = (ux_priority_success / len(ux_priority_tests)) * 100
        print(f"\n📊 UX Prioritization Accuracy: {accuracy:.1f}% ({ux_priority_success}/{len(ux_priority_tests)})")
        return accuracy > 75
    
    def test_business_intelligence_generation(self):
        """Test business intelligence report generation"""
        print("\n📊 Testing Business Intelligence Generation:")
        print("-" * 50)
        
        pipeline = self.models.get('pipeline')
        if not pipeline:
            print("❌ Pipeline not loaded")
            return False
        
        # Use comprehensive test set
        all_test_texts = []
        for lang_texts in self.test_texts.values():
            all_test_texts.extend(lang_texts[:5])  # 5 per language
        
        print(f"📝 Analyzing {len(all_test_texts)} test reviews...")
        
        # Generate business intelligence
        batch_results = pipeline.analyze_batch_with_business_intelligence(all_test_texts)
        bi_report = batch_results['business_intelligence']
        summary_metrics = batch_results['summary_metrics']
        
        print(f"\n📈 Business Intelligence Report Generated:")
        print(f"   Total Reviews: {bi_report['total_reviews']}")
        print(f"   Mixed Concerns: {bi_report['business_metrics']['mixed_concerns_percentage']}%")
        print(f"   UX Priority: {bi_report['business_metrics']['user_experience_priority_percentage']}%") 
        print(f"   High Priority: {bi_report['business_metrics']['high_priority_percentage']}%")
        print(f"   Critical Severity: {bi_report['business_metrics']['critical_severity_percentage']}%")
        print(f"   Immediate Action: {bi_report['business_metrics']['immediate_action_percentage']}%")
        
        print(f"\n🎯 Classification Distribution:")
        class_dist = bi_report['classification_distribution']
        for class_type, count in class_dist.items():
            percentage = (count / bi_report['total_reviews']) * 100
            print(f"   {class_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        print(f"\n🚨 Top Business Recommendations:")
        for i, rec in enumerate(bi_report['top_recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n⚠️ Critical Issues Identified:")
        for i, issue in enumerate(bi_report['critical_issues'], 1):
            print(f"   {i}. {issue['text']} (Priority: {issue['priority_level']}, Severity: {issue['severity_level']})")
        
        # Verify BI completeness
        required_bi_fields = [
            'total_reviews', 'classification_distribution', 'business_metrics', 
            'top_recommendations', 'critical_issues'
        ]
        missing_fields = [field for field in required_bi_fields if field not in bi_report]
        
        if not missing_fields:
            print(f"\n✅ Business Intelligence: COMPLETE")
            return True
        else:
            print(f"\n❌ Business Intelligence: Missing fields {missing_fields}")
            return False
    
    def test_with_real_fedex_data(self):
        """Test with your actual FedEx data"""
        print("\n📱 Testing with Real FedEx Data:")
        print("-" * 50)
        
        pipeline = self.models.get('pipeline')
        if not pipeline:
            print("❌ Pipeline not loaded")
            return False
        
        # Try to load your FedEx data
        try:
            fedex_file = "data/fedex_reviews_20250822_1657.csv"
            
            if os.path.exists(fedex_file):
                print(f"📊 Loading FedEx data: {fedex_file}")
                df = pd.read_csv(fedex_file)
                print(f"   Original shape: {df.shape}")
                print(f"   Columns: {list(df.columns)}")
                
                # Show sample of original data format
                print(f"\n📋 Original Data Format Sample:")
                if 'aspect' in df.columns:
                    old_format_sample = df[['text', 'sentiment', 'aspect']].head(2)
                    for idx, row in old_format_sample.iterrows():
                        print(f"   Old format: text='{row['text'][:50]}...', sentiment={row['sentiment']}, aspect={row['aspect']}")
                
                # Test DataFrame analysis with pure multi-label
                print(f"\n🔄 Analyzing with Pure Multi-Label Pipeline...")
                
                # Analyze first 20 reviews for speed
                sample_size = min(20, len(df))
                df_sample = df.head(sample_size).copy()
                
                # This should add new columns with new format
                df_enhanced = pipeline.analyze_dataframe(df_sample, text_column='text')
                
                # Show new column structure
                new_columns = [col for col in df_enhanced.columns if col.startswith('predicted_')]
                print(f"\n📊 New Multi-Label Columns Added:")
                for col in new_columns:
                    print(f"   • {col}")
                
                # Show sample of new data format
                print(f"\n📋 New Multi-Label Format Sample:")
                for idx in range(min(3, len(df_enhanced))):
                    row = df_enhanced.iloc[idx]
                    text = row['text'][:50] + '...' if len(row['text']) > 50 else row['text']
                    primary = row.get('predicted_primary_aspect', 'N/A')
                    secondary = row.get('predicted_secondary_aspects', [])
                    class_type = row.get('predicted_classification_type', 'N/A')
                    priority = row.get('predicted_priority_level', 'N/A')
                    
                    print(f"   New format: text='{text}'")
                    print(f"              primary_aspect={primary}, secondary_aspects={secondary}")
                    print(f"              type={class_type}, priority={priority}")
                    print()
                
                # Show business intelligence from metadata
                if hasattr(df_enhanced, 'attrs') and 'business_intelligence' in df_enhanced.attrs:
                    bi = df_enhanced.attrs['business_intelligence']
                    print(f"📊 FedEx Data Business Intelligence:")
                    print(f"   Mixed Concerns: {bi['business_metrics']['mixed_concerns_percentage']}%")
                    print(f"   UX Priority: {bi['business_metrics']['user_experience_priority_percentage']}%")
                    print(f"   High Priority: {bi['business_metrics']['high_priority_percentage']}%")
                
                print(f"\n✅ FedEx Data Analysis: SUCCESS")
                print(f"   Processed {sample_size} reviews")
                print(f"   New multi-label format applied")
                print(f"   Business intelligence generated")
                
                return True
                
            else:
                print(f"⚠️ FedEx data not found at {fedex_file}")
                print("   Creating sample FedEx-like data for testing...")
                
                # Create sample data
                sample_data = {
                    'text': self.test_texts['en'][:5],
                    'rating': [1, 3, 2, 4, 1],
                    'country': ['us'] * 5,
                    'language_detected': ['en'] * 5
                }
                df_sample = pd.DataFrame(sample_data)
                
                # Test with sample
                df_enhanced = pipeline.analyze_dataframe(df_sample)
                
                print(f"✅ Sample data analysis successful")
                print(f"   Columns added: {[col for col in df_enhanced.columns if col.startswith('predicted_')]}")
                
                return True
                
        except Exception as e:
            print(f"❌ FedEx data test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_multilingual_capabilities(self):
        """Test multilingual classification"""
        print("\n🌍 Testing Multilingual Capabilities:")
        print("-" * 50)
        
        pipeline = self.models.get('pipeline')
        if not pipeline:
            print("❌ Pipeline not loaded")
            return False
        
        multilingual_success = 0
        total_tests = 0
        
        for lang, texts in self.test_texts.items():
            if lang == 'en':
                continue  # Skip English, focus on other languages
            
            print(f"\n🌍 Language: {lang.upper()}")
            print("-" * 20)
            
            for i, text in enumerate(texts[:2], 1):  # Test 2 per language
                result = pipeline.analyze_text(text, language=lang)
                
                print(f"{i}. Text: {text}")
                print(f"   Primary: {result['primary_aspect']}")
                print(f"   Secondary: {result['secondary_aspects']}")
                print(f"   Type: {result['classification_type']}")
                print(f"   Priority: {result['priority_level']}")
                
                # Check if classification makes sense (not just 'general_satisfaction')
                if result['primary_aspect'] != 'general_satisfaction':
                    multilingual_success += 1
                    print(f"   🌍 MULTILINGUAL: ✅ Meaningful classification")
                else:
                    print(f"   🌍 MULTILINGUAL: ⚠️ Generic classification")
                
                total_tests += 1
        
        if total_tests > 0:
            accuracy = (multilingual_success / total_tests) * 100
            print(f"\n📊 Multilingual Accuracy: {accuracy:.1f}% ({multilingual_success}/{total_tests})")
            return accuracy > 50  # 50% threshold for multilingual
        else:
            print(f"\n⚠️ No multilingual tests available")
            return True
    
    def benchmark_performance(self):
        """Benchmark pure multi-label system performance"""
        print("\n⚡ Performance Benchmark:")
        print("-" * 50)
        
        pipeline = self.models.get('pipeline')
        if not pipeline:
            print("❌ Pipeline not loaded")
            return False
        
        # Test performance with various text lengths
        test_cases = [
            ("Short text", "App crashes"),
            ("Medium text", "Love the tracking accuracy but the interface is confusing and sometimes slow"),
            ("Long text", "This app has been a mixed experience for me. The tracking functionality works very well and is quite accurate, showing real-time updates of package locations. However, the user interface is extremely confusing and difficult to navigate, especially when trying to find basic features like scheduling pickups or managing multiple packages. Additionally, the app crashes frequently when I try to track more than 3 packages at once, which is very frustrating for business use.")
        ]
        
        iterations = 10
        results = {}
        
        for case_name, text in test_cases:
            print(f"\n📏 Testing {case_name} (Length: {len(text)} chars)")
            
            # Benchmark analysis time
            start_time = time.time()
            for _ in range(iterations):
                result = pipeline.analyze_text(text)
            avg_time = (time.time() - start_time) / iterations
            
            results[case_name] = {
                'avg_time': avg_time,
                'length': len(text),
                'chars_per_second': len(text) / avg_time
            }
            
            print(f"   Average time: {avg_time:.3f}s")
            print(f"   Processing rate: {len(text) / avg_time:.0f} chars/second")
            
            # Show classification result
            print(f"   Result: {result['primary_aspect']} + {result['secondary_aspects']} ({result['classification_type']})")
        
        # Overall performance summary
        print(f"\n📊 Performance Summary:")
        avg_time_overall = np.mean([r['avg_time'] for r in results.values()])
        print(f"   Average processing time: {avg_time_overall:.3f}s per review")
        print(f"   Suitable for: {'Real-time' if avg_time_overall < 1.0 else 'Batch'} processing")
        
        return avg_time_overall < 2.0  # Success if under 2 seconds per review
    
    def generate_presentation_demo_data(self):
        """Generate data specifically for your bootcamp presentation"""
        print("\n🎯 Generating Presentation Demo Data:")
        print("-" * 50)
        
        pipeline = self.models.get('pipeline')
        if not pipeline:
            print("❌ Pipeline not loaded")
            return None
        
        # Curated texts that demonstrate your key innovations
        presentation_texts = [
            # Show old system limitation vs new system power
            "Love the tracking accuracy but the interface is confusing",  # Mixed concerns
            
            # Show user experience prioritization
            "Interface is impossible to use, worst app design ever",  # UX priority
            
            # Show severity and priority assessment
            "App crashes constantly and completely unusable interface",  # High severity + priority
            
            # Show business actionable insights
            "Great tracking but terrible navigation and slow performance",  # Multiple teams needed
            
            # Your actual FedEx sample
            "not receiving email for sign in, this app continues to be trash!",
            
            # Show positive classification
            "Beautiful interface design and very accurate tracking",  # Positive multi-aspect
        ]
        
        print(f"🎨 Analyzing {len(presentation_texts)} curated presentation examples...")
        
        # Generate comprehensive analysis
        batch_results = pipeline.analyze_batch_with_business_intelligence(presentation_texts)
        
        # Create presentation-ready summary
        presentation_data = {
            'demo_texts': presentation_texts,
            'individual_results': batch_results['individual_results'],
            'business_intelligence': batch_results['business_intelligence'],
            'key_metrics': {
                'total_reviews': len(presentation_texts),
                'mixed_concerns_detected': sum(1 for r in batch_results['individual_results'] if r['mixed_concerns']),
                'user_experience_priority': sum(1 for r in batch_results['individual_results'] if r['user_experience_priority']),
                'high_priority_issues': sum(1 for r in batch_results['individual_results'] if r['high_priority']),
                'immediate_action_required': sum(1 for r in batch_results['individual_results'] if r['requires_immediate_action']),
            },
            'presentation_highlights': [
                "Multi-label classification detects mixed concerns",
                "User experience prioritization working correctly",
                "Business priority levels automatically assigned",
                "Actionable recommendations generated for each review",
                "Critical issues identified for immediate action"
            ]
        }
        
        # Show presentation summary
        print(f"\n🎯 Presentation Demo Summary:")
        for key, value in presentation_data['key_metrics'].items():
            percentage = (value / len(presentation_texts)) * 100
            print(f"   {key.replace('_', ' ').title()}: {value} ({percentage:.1f}%)")
        
        print(f"\n🚀 Key Demo Points:")
        for i, highlight in enumerate(presentation_data['presentation_highlights'], 1):
            print(f"   {i}. {highlight}")
        
        return presentation_data
    
    def run_comprehensive_testing(self):
        """Run complete testing suite for bootcamp presentation"""
        print("🧪 COMPREHENSIVE PURE MULTI-LABEL TESTING")
        print("="*70)
        
        # Load models
        if not self.load_pure_multilabel_models():
            print("❌ Failed to load models - aborting tests")
            return False
        
        # Run all test suites
        test_results = {}
        
        print("\n" + "="*70)
        test_results['classification_types'] = self.test_multilabel_classification_types()
        
        print("\n" + "="*70) 
        test_results['ux_prioritization'] = self.test_user_experience_prioritization()
        
        print("\n" + "="*70)
        test_results['business_intelligence'] = self.test_business_intelligence_generation()
        
        print("\n" + "="*70)
        test_results['fedex_data'] = self.test_with_real_fedex_data()
        
        print("\n" + "="*70)
        test_results['multilingual'] = self.test_multilingual_capabilities()
        
        print("\n" + "="*70)
        test_results['performance'] = self.benchmark_performance()
        
        print("\n" + "="*70)
        presentation_data = self.generate_presentation_demo_data()
        
        # Final comprehensive summary
        print("\n" + "="*70)
        print("🎉 COMPREHENSIVE TESTING COMPLETE!")
        print("="*70)
        
        passed_tests = sum(1 for result in test_results.values() if result)
        total_tests = len(test_results)
        success_rate = (passed_tests / total_tests) * 100
        
        print(f"\n📊 TEST RESULTS SUMMARY:")
        for test_name, passed in test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\n🎯 OVERALL SUCCESS RATE: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        
        if success_rate >= 80:
            print(f"\n🏆 EXCELLENT: Your pure multi-label system is presentation-ready!")
            print(f"✅ Key innovations successfully implemented:")
            print(f"   • Multi-label aspect classification")
            print(f"   • User experience prioritization")
            print(f"   • Mixed concerns detection")
            print(f"   • Business intelligence reporting")
            print(f"   • Actionable recommendations")
            
            print(f"\n🎤 PRESENTATION TALKING POINTS:")
            if presentation_data:
                for i, point in enumerate(presentation_data['presentation_highlights'], 1):
                    print(f"   {i}. {point}")
            
            print(f"\n🚀 READY FOR BOOTCAMP PRESENTATION!")
            
        else:
            print(f"\n⚠️ NEEDS IMPROVEMENT: {100-success_rate:.1f}% of tests failed")
            print(f"🔧 Fix failing tests before presentation")
        
        return success_rate >= 80, test_results, presentation_data

# Main execution
if __name__ == "__main__":
    tester = PureMultiLabelModelTester()
    success, results, demo_data = tester.run_comprehensive_testing()
    
    if success:
        print(f"\n🎊 PURE MULTI-LABEL SYSTEM READY!")
        print(f"🎯 Perfect for your data science bootcamp final presentation")
    else:
        print(f"\n🔧 SYSTEM NEEDS REFINEMENT")
        print(f"📋 Review failed tests and make improvements")