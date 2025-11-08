#!/usr/bin/env python3
"""
Pure Multi-Label Integrated ML Pipeline - UPDATED for Two-Model Ensemble
Save as: src/integrated_ml_pipeline.py

Updated to work with the new two-model ensemble sentiment classifier
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import time
import logging
from datetime import datetime
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.models.enhanced_sentiment_classifier import EnhancedSentimentClassifier
    from src.models.enhanced_aspect_classifier import EnhancedAspectClassifier
    ENHANCED_MODELS_AVAILABLE = True
    print("Enhanced models with two-model ensemble imported successfully")
except ImportError:
    print("Could not import enhanced models, using fallback")
    EnhancedSentimentClassifier = None
    EnhancedAspectClassifier = None
    ENHANCED_MODELS_AVAILABLE = False

class IntegratedMLPipeline:
    """
    Pure Multi-Label Integrated ML Pipeline for FedEx Review Analysis
    Updated to work with two-model ensemble sentiment classifier
    """
    
    def __init__(self, device='auto', verbose=True):
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        self._initialize_classifiers(device)
    
    def _initialize_classifiers(self, device='auto'):
        """Initialize enhanced classifiers with two-model ensemble support"""
        try:
            if ENHANCED_MODELS_AVAILABLE:
                if self.verbose:
                    print("Initializing Two-Model Ensemble ML Pipeline...")
                
                # Initialize sentiment classifier with two-model ensemble
                self.sentiment_classifier = EnhancedSentimentClassifier(
                    device=device,
                    verbose=self.verbose
                )
                
                # Initialize aspect classifier
                self.aspect_classifier = EnhancedAspectClassifier()
                
                if self.verbose:
                    print("Pure multi-label pipeline with two-model ensemble initialized")
            else:
                self._initialize_fallback_models()
            
            self.logger.info("Pure multi-label pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self):
        """Initialize fallback models when enhanced models aren't available"""
        self.sentiment_classifier = None
        self.aspect_classifier = None
        
        # Fallback sentiment analysis
        try:
            from transformers import pipeline
            self.sentiment_pipeline = pipeline(
                'sentiment-analysis', 
                model='nlptown/bert-base-multilingual-uncased-sentiment',
                return_all_scores=True
            )
            if self.verbose:
                print("Fallback sentiment classifier loaded")
        except:
            self.sentiment_pipeline = None
        
        # Fallback aspect keywords
        self.aspect_keywords = {
            'user_experience': ['easy', 'difficult', 'interface', 'design', 'intuitive', 'confusing', 'simple'],
            'performance': ['quality', 'performance', 'build', 'durable', 'reliable', 'fast', 'crash', 'bug'],
            'tracking_accuracy': ['tracking', 'location', 'status', 'updates', 'accurate', 'wrong'],
            'delivery_issues': ['delivery', 'arrive', 'shipping', 'package', 'late', 'on time'],
            'interface_design': ['design', 'look', 'appearance', 'visual', 'beautiful', 'ugly'],
            'general_satisfaction': ['overall', 'general', 'satisfied', 'recommend', 'love', 'hate']
        }
        
        if self.verbose:
            print("Fallback models initialized")
    
    def analyze_text(self, text: str, language: str = 'auto') -> Dict:
        """
        MAIN METHOD: Analyze single text with pure multi-label classification
        Uses two-model ensemble for sentiment analysis
        
        Returns new format: primary_aspect, secondary_aspects, etc.
        """
        start_time = time.time()
        
        # Sentiment analysis with two-model ensemble
        if self.sentiment_classifier:
            sentiment_result = self.sentiment_classifier.analyze_sentiment(text, language)
        else:
            sentiment_result = self._fallback_sentiment_analysis(text)
        
        # Multi-label aspect analysis with sentiment context
        if self.aspect_classifier:
            aspect_result = self.aspect_classifier.classify_aspects_multilabel(
                text, 
                language,
                sentiment_result['sentiment'],
                sentiment_result['confidence']
            )
        else:
            aspect_result = self._fallback_multilabel_aspect_analysis(text)
        
        # NEW FORMAT - Enhanced with two-model ensemble metadata
        return {
            'text': text,
            'language': language,
            
            # Sentiment (with two-model ensemble info)
            'sentiment': sentiment_result['sentiment'],
            'sentiment_confidence': sentiment_result['confidence'],
            'sentiment_method': sentiment_result.get('method', 'unknown'),
            'sentiment_models_used': sentiment_result.get('models_used', 0),
            'sentiment_device': sentiment_result.get('device', 'unknown'),
            'sentiment_from_cache': sentiment_result.get('from_cache', False),
            
            # NEW: Multi-label aspects
            'primary_aspect': aspect_result['primary_aspect'],
            'secondary_aspects': aspect_result['secondary_aspects'],
            'classification_type': aspect_result['classification_type'],
            'confidence': aspect_result['confidence'],
            'priority_level': aspect_result['priority_level'],
            'severity_level': aspect_result['severity_level'],
            'business_summary': aspect_result['business_summary'],
            'recommendation': aspect_result['recommendation'],
            'requires_immediate_action': aspect_result['requires_immediate_action'],
            
            # Business intelligence flags
            'all_aspect_scores': aspect_result.get('all_scores', {}),
            'user_experience_priority': aspect_result['primary_aspect'] == 'user_experience',
            'mixed_concerns': aspect_result['classification_type'] == 'mixed_concerns',
            'high_priority': aspect_result['priority_level'] == 'HIGH',
            'critical_severity': aspect_result['severity_level'] == 'CRITICAL',
            
            # System info (enhanced with ensemble data)
            'processing_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': '2.0_two_model_ensemble'
        }
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict:
        """Fallback sentiment analysis"""
        try:
            if self.sentiment_pipeline:
                prediction = self.sentiment_pipeline(text)
                scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
                
                if isinstance(prediction, list) and len(prediction) > 0:
                    if isinstance(prediction[0], list):
                        prediction = prediction[0]
                    
                    for item in prediction:
                        label = item['label'].lower()
                        score = item['score']
                        
                        if any(x in label for x in ['pos', '4', '5']):
                            scores['positive'] += score
                        elif any(x in label for x in ['neg', '1', '2']):
                            scores['negative'] += score
                        else:
                            scores['neutral'] += score
                
                max_sentiment = max(scores, key=scores.get)
                confidence = scores[max_sentiment]
                
                return {
                    'sentiment': max_sentiment, 
                    'confidence': confidence, 
                    'scores': scores,
                    'method': 'fallback_bert',
                    'models_used': 1,
                    'device': 'unknown',
                    'from_cache': False
                }
            else:
                return {
                    'sentiment': 'neutral', 
                    'confidence': 0.5, 
                    'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                    'method': 'basic_fallback',
                    'models_used': 0,
                    'device': 'cpu',
                    'from_cache': False
                }
            
        except Exception as e:
            return {
                'sentiment': 'neutral', 
                'confidence': 0.5, 
                'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'method': 'error_fallback',
                'models_used': 0,
                'device': 'unknown',
                'from_cache': False
            }
    
    def _fallback_multilabel_aspect_analysis(self, text: str) -> Dict:
        """Fallback multi-label aspect analysis"""
        text_lower = text.lower()
        scores = {}
        
        # Score all aspects
        for aspect, keywords in self.aspect_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[aspect] = score
        
        # Find significant aspects
        max_score = max(scores.values()) if scores else 0
        if max_score == 0:
            return {
                'primary_aspect': 'general_satisfaction',
                'secondary_aspects': [],
                'classification_type': 'unclear',
                'confidence': 0.5,
                'priority_level': 'LOW',
                'severity_level': 'MODERATE',
                'business_summary': 'No clear aspect detected',
                'recommendation': 'Review manually for proper categorization',
                'requires_immediate_action': False,
                'all_scores': scores
            }
        
        # Sort by score
        sorted_aspects = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        significant_aspects = [(aspect, score) for aspect, score in sorted_aspects if score > 0]
        
        # Determine classification
        primary_aspect = significant_aspects[0][0]
        secondary_aspects = [aspect for aspect, score in significant_aspects[1:3] if score > 0]
        
        classification_type = (
            'single_aspect' if len(secondary_aspects) == 0
            else 'dual_aspect' if len(secondary_aspects) == 1
            else 'mixed_concerns'
        )
        
        # Simple priority logic
        priority_level = 'HIGH' if primary_aspect in ['user_experience', 'performance'] else 'MEDIUM'
        
        return {
            'primary_aspect': primary_aspect,
            'secondary_aspects': secondary_aspects,
            'classification_type': classification_type,
            'confidence': max_score,
            'priority_level': priority_level,
            'severity_level': 'MODERATE',
            'business_summary': f"{classification_type.replace('_', ' ').title()} - {primary_aspect.replace('_', ' ')}",
            'recommendation': f"Route to appropriate team for {primary_aspect.replace('_', ' ')}",
            'requires_immediate_action': priority_level == 'HIGH' and len(secondary_aspects) > 0,
            'all_scores': scores
        }
    
    def analyze_batch(self, texts: List[str], languages: List[str] = None) -> List[Dict]:
        """
        Analyze multiple texts with multi-label classification
        Optimized for two-model ensemble processing
        """
        if languages is None:
            languages = ['auto'] * len(texts)
        
        results = []
        
        if self.verbose and len(texts) > 10:
            print(f"Processing batch of {len(texts)} texts with two-model ensemble...")
        
        # Use optimized batch processing if available
        if self.sentiment_classifier and hasattr(self.sentiment_classifier, 'analyze_batch'):
            # Get all sentiment results in one batch (more efficient)
            sentiment_results = self.sentiment_classifier.analyze_batch(texts)
            
            # Process aspects with sentiment context
            for i, (text, lang, sentiment_result) in enumerate(zip(texts, languages, sentiment_results)):
                if self.aspect_classifier:
                    aspect_result = self.aspect_classifier.classify_aspects_multilabel(
                        text, 
                        lang,
                        sentiment_result['sentiment'],
                        sentiment_result['confidence']
                    )
                else:
                    aspect_result = self._fallback_multilabel_aspect_analysis(text)
                
                # Combine results
                combined_result = self._combine_sentiment_aspect_results(
                    text, lang, sentiment_result, aspect_result
                )
                results.append(combined_result)
                
                # Progress update for large batches
                if self.verbose and len(texts) > 100 and (i + 1) % 50 == 0:
                    print(f"  Progress: {i + 1}/{len(texts)} ({(i + 1)/len(texts)*100:.1f}%)")
        else:
            # Fallback to individual processing
            for text, lang in zip(texts, languages):
                result = self.analyze_text(text, lang)
                results.append(result)
        
        return results
    
    def _combine_sentiment_aspect_results(self, text: str, language: str, 
                                        sentiment_result: Dict, aspect_result: Dict) -> Dict:
        """Combine sentiment and aspect results into final format"""
        start_time = time.time()
        
        return {
            'text': text,
            'language': language,
            
            # Sentiment (with two-model ensemble info)
            'sentiment': sentiment_result['sentiment'],
            'sentiment_confidence': sentiment_result['confidence'],
            'sentiment_method': sentiment_result.get('method', 'unknown'),
            'sentiment_models_used': sentiment_result.get('models_used', 0),
            'sentiment_device': sentiment_result.get('device', 'unknown'),
            'sentiment_from_cache': sentiment_result.get('from_cache', False),
            
            # Multi-label aspects
            'primary_aspect': aspect_result['primary_aspect'],
            'secondary_aspects': aspect_result['secondary_aspects'],
            'classification_type': aspect_result['classification_type'],
            'confidence': aspect_result['confidence'],
            'priority_level': aspect_result['priority_level'],
            'severity_level': aspect_result['severity_level'],
            'business_summary': aspect_result['business_summary'],
            'recommendation': aspect_result['recommendation'],
            'requires_immediate_action': aspect_result['requires_immediate_action'],
            
            # Business intelligence flags
            'all_aspect_scores': aspect_result.get('all_scores', {}),
            'user_experience_priority': aspect_result['primary_aspect'] == 'user_experience',
            'mixed_concerns': aspect_result['classification_type'] == 'mixed_concerns',
            'high_priority': aspect_result['priority_level'] == 'HIGH',
            'critical_severity': aspect_result['severity_level'] == 'CRITICAL',
            
            # System info
            'processing_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': '2.0_two_model_ensemble'
        }
    
    def analyze_batch_with_business_intelligence(self, texts: List[str], languages: List[str] = None) -> Dict:
        """
        Analyze batch with comprehensive business intelligence reporting
        Enhanced with two-model ensemble performance metrics
        """
        results = self.analyze_batch(texts, languages)
        business_report = self._generate_business_intelligence(results)
        
        return {
            'individual_results': results,
            'business_intelligence': business_report,
            'summary_metrics': self._generate_summary_metrics(results),
            'ensemble_performance': self._generate_ensemble_metrics(results)
        }
    
    def _generate_business_intelligence(self, results: List[Dict]) -> Dict:
        """Generate comprehensive business intelligence report"""
        if not results:
            return {}
        
        total = len(results)
        
        # Classification type distribution
        classification_counts = {}
        for result in results:
            class_type = result.get('classification_type', 'single_aspect')
            classification_counts[class_type] = classification_counts.get(class_type, 0) + 1
        
        # Priority and severity distributions
        priority_counts = {}
        severity_counts = {}
        primary_aspect_counts = {}
        
        # Business metrics
        mixed_concerns_count = 0
        ux_priority_count = 0
        high_priority_count = 0
        critical_severity_count = 0
        immediate_action_count = 0
        
        for result in results:
            # Priority level
            priority = result.get('priority_level', 'MEDIUM')
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            # Severity level
            severity = result.get('severity_level', 'MODERATE')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Primary aspect
            primary = result.get('primary_aspect', 'general_satisfaction')
            primary_aspect_counts[primary] = primary_aspect_counts.get(primary, 0) + 1
            
            # Business flags
            if result.get('mixed_concerns', False):
                mixed_concerns_count += 1
            if result.get('user_experience_priority', False):
                ux_priority_count += 1
            if result.get('high_priority', False):
                high_priority_count += 1
            if result.get('critical_severity', False):
                critical_severity_count += 1
            if result.get('requires_immediate_action', False):
                immediate_action_count += 1
        
        return {
            'total_reviews': total,
            'classification_distribution': {
                'single_aspect': classification_counts.get('single_aspect', 0),
                'dual_aspect': classification_counts.get('dual_aspect', 0),
                'mixed_concerns': classification_counts.get('mixed_concerns', 0)
            },
            'priority_distribution': priority_counts,
            'severity_distribution': severity_counts,
            'aspect_distribution': primary_aspect_counts,
            'business_metrics': {
                'mixed_concerns_percentage': round((mixed_concerns_count / total) * 100, 1),
                'user_experience_priority_percentage': round((ux_priority_count / total) * 100, 1),
                'high_priority_percentage': round((high_priority_count / total) * 100, 1),
                'critical_severity_percentage': round((critical_severity_count / total) * 100, 1),
                'immediate_action_percentage': round((immediate_action_count / total) * 100, 1)
            },
            'top_recommendations': self._generate_top_business_recommendations(results),
            'critical_issues': self._identify_critical_issues(results)
        }
    
    def _generate_ensemble_metrics(self, results: List[Dict]) -> Dict:
        """Generate two-model ensemble performance metrics"""
        if not results:
            return {}
        
        total = len(results)
        
        # Count by sentiment method
        method_counts = {}
        models_used_counts = {}
        device_counts = {}
        cache_hits = 0
        
        processing_times = []
        
        for result in results:
            # Sentiment method
            method = result.get('sentiment_method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
            
            # Models used
            models_used = result.get('sentiment_models_used', 0)
            models_used_counts[models_used] = models_used_counts.get(models_used, 0) + 1
            
            # Device usage
            device = result.get('sentiment_device', 'unknown')
            device_counts[device] = device_counts.get(device, 0) + 1
            
            # Cache performance
            if result.get('sentiment_from_cache', False):
                cache_hits += 1
            
            # Processing time
            processing_times.append(result.get('processing_time', 0))
        
        return {
            'total_analyzed': total,
            'sentiment_method_distribution': method_counts,
            'models_used_distribution': models_used_counts,
            'device_usage': device_counts,
            'cache_performance': {
                'hits': cache_hits,
                'hit_rate_percentage': round((cache_hits / total) * 100, 1),
                'misses': total - cache_hits
            },
            'performance_metrics': {
                'average_processing_time_ms': round(np.mean(processing_times) * 1000, 2),
                'total_processing_time_s': round(sum(processing_times), 2),
                'throughput_texts_per_second': round(total / sum(processing_times), 2) if sum(processing_times) > 0 else 0
            },
            'ensemble_efficiency': {
                'two_model_ensemble_usage_pct': round(
                    (method_counts.get('two_model_ensemble', 0) / total) * 100, 1
                ),
                'fallback_usage_pct': round(
                    (method_counts.get('rule_based', 0) / total) * 100, 1
                )
            }
        }
    
    def _generate_top_business_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate top business recommendations"""
        recommendations = []
        
        # Count critical and high priority issues
        critical_count = sum(1 for r in results if r.get('critical_severity', False))
        high_priority_count = sum(1 for r in results if r.get('high_priority', False))
        immediate_action_count = sum(1 for r in results if r.get('requires_immediate_action', False))
        
        if critical_count > 0:
            recommendations.append(f"CRITICAL: {critical_count} reviews with critical severity need immediate attention")
        
        if immediate_action_count > 0:
            recommendations.append(f"URGENT: {immediate_action_count} reviews require immediate action")
        
        if high_priority_count > len(results) * 0.3:
            recommendations.append(f"HIGH VOLUME: {high_priority_count} high-priority issues detected ({(high_priority_count/len(results)*100):.1f}%)")
        
        # Most common high priority aspects
        high_priority_reviews = [r for r in results if r.get('high_priority', False)]
        if high_priority_reviews:
            from collections import Counter
            common_aspects = Counter([r['primary_aspect'] for r in high_priority_reviews]).most_common(2)
            for aspect, count in common_aspects:
                recommendations.append(f"Focus on {aspect.replace('_', ' ')}: {count} high-priority cases")
        
        return recommendations[:5]
    
    def _identify_critical_issues(self, results: List[Dict]) -> List[Dict]:
        """Identify most critical reviews"""
        critical_reviews = []
        
        for result in results:
            if (result.get('critical_severity', False) or 
                result.get('requires_immediate_action', False) or
                (result.get('high_priority', False) and result.get('mixed_concerns', False))):
                
                critical_reviews.append({
                    'text': result['text'][:100] + '...' if len(result['text']) > 100 else result['text'],
                    'primary_aspect': result['primary_aspect'],
                    'secondary_aspects': result['secondary_aspects'],
                    'priority_level': result['priority_level'],
                    'severity_level': result['severity_level'],
                    'classification_type': result['classification_type'],
                    'confidence': result['confidence'],
                    'recommendation': result['recommendation'],
                    'sentiment_method': result.get('sentiment_method', 'unknown')
                })
        
        # Sort by confidence and return top 5
        return sorted(critical_reviews, key=lambda x: x['confidence'], reverse=True)[:5]
    
    def _generate_summary_metrics(self, results: List[Dict]) -> Dict:
        """Generate summary metrics for dashboard"""
        if not results:
            return {}
        
        total = len(results)
        
        # Sentiment distribution
        sentiment_counts = {}
        for result in results:
            sentiment = result.get('sentiment', 'neutral')
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        # Average confidences
        sentiment_confidences = [r.get('sentiment_confidence', 0) for r in results]
        aspect_confidences = [r.get('confidence', 0) for r in results]
        processing_times = [r.get('processing_time', 0) for r in results]
        
        return {
            'total_analyzed': total,
            'sentiment_distribution': sentiment_counts,
            'average_sentiment_confidence': round(np.mean(sentiment_confidences), 3),
            'average_aspect_confidence': round(np.mean(aspect_confidences), 3),
            'average_processing_time': round(np.mean(processing_times), 3),
            'multilabel_features': True,
            'two_model_ensemble': ENHANCED_MODELS_AVAILABLE and self.sentiment_classifier is not None
        }
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Analyze DataFrame with pure multi-label classification
        Enhanced with two-model ensemble processing
        
        Returns DataFrame with new column structure including ensemble metadata
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        print(f"Analyzing {len(df)} reviews with pure multi-label classification + two-model ensemble...")
        
        texts = df[text_column].astype(str).tolist()
        
        # Analyze with business intelligence
        batch_results = self.analyze_batch_with_business_intelligence(texts)
        results = batch_results['individual_results']
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        output_df = df.copy()
        
        # Add new multi-label columns including ensemble metadata
        new_columns = [
            'sentiment', 'sentiment_confidence', 'sentiment_method', 'sentiment_models_used', 'sentiment_from_cache',
            'primary_aspect', 'secondary_aspects', 'classification_type', 
            'confidence', 'priority_level', 'severity_level',
            'business_summary', 'recommendation', 'requires_immediate_action',
            'user_experience_priority', 'mixed_concerns', 'high_priority', 'critical_severity'
        ]
        
        for col in new_columns:
            if col in results_df.columns:
                output_df[f'predicted_{col}'] = results_df[col]
        
        print(f"Added {len([col for col in new_columns if col in results_df.columns])} new multi-label columns")
        
        # Add enhanced metadata
        output_df.attrs['business_intelligence'] = batch_results['business_intelligence']
        output_df.attrs['summary_metrics'] = batch_results['summary_metrics']
        output_df.attrs['ensemble_performance'] = batch_results['ensemble_performance']
        
        return output_df
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the pipeline configuration"""
        info = {
            'pipeline_type': 'Pure Multi-Label with Two-Model Ensemble',
            'version': '2.0_two_model_ensemble',
            'backward_compatible': False,
            'features': [
                'Multi-label aspect classification',
                'Two-model ensemble sentiment analysis',
                'User experience prioritization', 
                'Mixed concerns detection',
                'Business priority levels',
                'Severity assessment',
                'Actionable recommendations',
                'Business intelligence reporting',
                'Ensemble performance metrics',
                'Cache optimization'
            ],
            'output_format': 'New multi-label format with primary_aspect + secondary_aspects + ensemble metadata',
            'suitable_for': 'Advanced data science projects and production deployments',
            'models_loaded': {
                'enhanced_sentiment_classifier': ENHANCED_MODELS_AVAILABLE and self.sentiment_classifier is not None,
                'enhanced_aspect_classifier': ENHANCED_MODELS_AVAILABLE and self.aspect_classifier is not None
            }
        }
        
        # Add sentiment classifier details if available
        if ENHANCED_MODELS_AVAILABLE and self.sentiment_classifier:
            try:
                sentiment_info = self.sentiment_classifier.get_model_info()
                info['sentiment_classifier_details'] = sentiment_info
            except:
                info['sentiment_classifier_details'] = 'Available but details unavailable'
        
        return info
    
    def cleanup(self):
        """Clean up pipeline resources"""
        if self.sentiment_classifier and hasattr(self.sentiment_classifier, 'cleanup'):
            self.sentiment_classifier.cleanup()
        
        if self.verbose:
            print("Pipeline cleanup completed")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Pure Multi-Label Integrated Pipeline with Two-Model Ensemble")
    print("="*70)
    
    # Initialize pipeline
    pipeline = IntegratedMLPipeline(device='auto', verbose=True)
    
    # Test with your FedEx sample
    test_texts = [
        "not receiving email for sign in, this app continues to be trash!",
        "Love the tracking but the interface is confusing",
        "App crashes constantly when trying to track packages",
        "Great delivery notifications but app design is ugly and hard to navigate"
    ]
    
    print("\nTesting Individual Text Analysis:")
    print("-" * 50)
    
    for i, text in enumerate(test_texts, 1):
        result = pipeline.analyze_text(text)
        
        print(f"\n{i}. Text: {text}")
        print(f"   Sentiment: {result['sentiment']} ({result['sentiment_confidence']:.3f})")
        print(f"   Sentiment Method: {result['sentiment_method']}")
        print(f"   Models Used: {result['sentiment_models_used']}")
        print(f"   From Cache: {result['sentiment_from_cache']}")
        print(f"   Device: {result['sentiment_device']}")
        print(f"   Primary Aspect: {result['primary_aspect']}")
        print(f"   Secondary Aspects: {result['secondary_aspects']}")
        print(f"   Classification Type: {result['classification_type']}")
        print(f"   Priority: {result['priority_level']}")
        print(f"   Severity: {result['severity_level']}")
        print(f"   Immediate Action: {result['requires_immediate_action']}")
        print(f"   Business Summary: {result['business_summary']}")
    
    print(f"\nTesting Batch Analysis with Business Intelligence:")
    print("-" * 50)
    
    # Test batch analysis
    batch_results = pipeline.analyze_batch_with_business_intelligence(test_texts)
    bi = batch_results['business_intelligence']
    ensemble_perf = batch_results['ensemble_performance']
    
    print(f"\nBusiness Intelligence Report:")
    print(f"   Total Reviews: {bi['total_reviews']}")
    print(f"   Mixed Concerns: {bi['business_metrics']['mixed_concerns_percentage']}%")
    print(f"   UX Priority: {bi['business_metrics']['user_experience_priority_percentage']}%")
    print(f"   High Priority: {bi['business_metrics']['high_priority_percentage']}%")
    print(f"   Critical Severity: {bi['business_metrics']['critical_severity_percentage']}%")
    print(f"   Immediate Action: {bi['business_metrics']['immediate_action_percentage']}%")
    
    print(f"\nTwo-Model Ensemble Performance:")
    print(f"   Methods Used: {ensemble_perf['sentiment_method_distribution']}")
    print(f"   Models Distribution: {ensemble_perf['models_used_distribution']}")
    print(f"   Cache Hit Rate: {ensemble_perf['cache_performance']['hit_rate_percentage']}%")
    print(f"   Average Processing Time: {ensemble_perf['performance_metrics']['average_processing_time_ms']:.1f}ms")
    print(f"   Throughput: {ensemble_perf['performance_metrics']['throughput_texts_per_second']:.1f} texts/sec")
    print(f"   Ensemble Usage: {ensemble_perf['ensemble_efficiency']['two_model_ensemble_usage_pct']}%")
    
    print(f"\nTop Recommendations:")
    for rec in bi['top_recommendations']:
        print(f"   • {rec}")
    
    print(f"\nCritical Issues:")
    for i, issue in enumerate(bi['critical_issues'], 1):
        print(f"   {i}. {issue['text']} (Priority: {issue['priority_level']}, Method: {issue['sentiment_method']})")
    
    # Show pipeline info
    print(f"\nPipeline Information:")
    info = pipeline.get_pipeline_info()
    for key, value in info.items():
        if isinstance(value, list):
            print(f"   {key.replace('_', ' ').title()}:")
            for item in value:
                print(f"      • {item}")
        elif isinstance(value, dict):
            print(f"   {key.replace('_', ' ').title()}:")
            for sub_key, sub_value in value.items():
                print(f"      {sub_key}: {sub_value}")
        else:
            print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\nPure Multi-Label Pipeline with Two-Model Ensemble Ready!")
    print(f"Perfect for your bootcamp presentation")
    print(f"Shows advanced ML concepts including ensemble methods")