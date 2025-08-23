#!/usr/bin/env python3
"""
Multi-Label Integrated ML Pipeline - FINAL VERSION

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
    from models.enhanced_sentiment_classifier import EnhancedSentimentClassifier
    from models.enhanced_aspect_classifier import EnhancedAspectClassifier
except ImportError:
    print("⚠️ Could not import enhanced models, using fallback")
    EnhancedSentimentClassifier = None
    EnhancedAspectClassifier = None

class IntegratedMLPipeline:
    """Pure Multi-Label Integrated ML Pipeline for FedEx Review Analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_classifiers()
    
    def _initialize_classifiers(self):
        """Initialize enhanced classifiers"""
        try:
            if EnhancedSentimentClassifier:
                self.sentiment_classifier = EnhancedSentimentClassifier()
            else:
                self._initialize_fallback_sentiment()
            
            if EnhancedAspectClassifier:
                self.aspect_classifier = EnhancedAspectClassifier()
                print("✅ Pure multi-label aspect classifier loaded")
            else:
                self._initialize_fallback_aspect()
            
            self.logger.info("✅ Pure multi-label pipeline initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            self._initialize_fallback_models()
    
    def _initialize_fallback_sentiment(self):
        """Initialize fallback sentiment analysis"""
        from transformers import pipeline
        
        self.sentiment_pipeline = pipeline('sentiment-analysis', 
                                         model='nlptown/bert-base-multilingual-uncased-sentiment',
                                         return_all_scores=True)
        self.logger.info("✅ Fallback sentiment classifier loaded")
    
    def _initialize_fallback_aspect(self):
        """Initialize fallback aspect detection"""
        self.aspect_keywords = {
            'user_experience': ['easy', 'difficult', 'interface', 'design', 'intuitive', 'confusing', 'simple'],
            'performance': ['quality', 'performance', 'build', 'durable', 'reliable', 'fast', 'crash', 'bug'],
            'tracking_accuracy': ['tracking', 'location', 'status', 'updates', 'accurate', 'wrong'],
            'delivery_issues': ['delivery', 'arrive', 'shipping', 'package', 'late', 'on time'],
            'interface_design': ['design', 'look', 'appearance', 'visual', 'beautiful', 'ugly'],
            'general_satisfaction': ['overall', 'general', 'satisfied', 'recommend', 'love', 'hate']
        }
        self.logger.info("✅ Fallback aspect classifier loaded")
    
    def _initialize_fallback_models(self):
        """Initialize minimal fallback models"""
        self._initialize_fallback_sentiment()
        self._initialize_fallback_aspect()
    
    def analyze_text(self, text: str, language: str = 'auto') -> Dict:
        """
        MAIN METHOD: Analyze single text with pure multi-label classification
        
        Returns new format: primary_aspect, secondary_aspects, etc.
        """
        start_time = time.time()
        
        # Sentiment analysis
        if hasattr(self, 'sentiment_classifier'):
            sentiment_result = self.sentiment_classifier.analyze_sentiment(text, language)
        else:
            sentiment_result = self._fallback_sentiment_analysis(text)
        
        # Multi-label aspect analysis
        if hasattr(self, 'aspect_classifier'):
            aspect_result = self.aspect_classifier.classify_aspects_multilabel(text, language)
        else:
            aspect_result = self._fallback_multilabel_aspect_analysis(text)
        
        # NEW FORMAT - No backward compatibility
        return {
            'text': text,
            'language': language,
            
            # Sentiment (unchanged)
            'sentiment': sentiment_result['sentiment'],
            'sentiment_confidence': sentiment_result['confidence'],
            
            # NEW: Multi-label aspects (replaces old 'aspect' field)
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
            'timestamp': datetime.now().isoformat()
        }
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict:
        """Fallback sentiment analysis"""
        try:
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
            
            return {'sentiment': max_sentiment, 'confidence': confidence, 'scores': scores}
            
        except Exception as e:
            return {'sentiment': 'neutral', 'confidence': 0.5, 'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}}
    
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
        """Analyze multiple texts with multi-label classification"""
        if languages is None:
            languages = ['auto'] * len(texts)
        
        results = []
        for text, lang in zip(texts, languages):
            result = self.analyze_text(text, lang)
            results.append(result)
        
        return results
    
    def analyze_batch_with_business_intelligence(self, texts: List[str], languages: List[str] = None) -> Dict:
        """Analyze batch with comprehensive business intelligence reporting"""
        results = self.analyze_batch(texts, languages)
        business_report = self._generate_business_intelligence(results)
        
        return {
            'individual_results': results,
            'business_intelligence': business_report,
            'summary_metrics': self._generate_summary_metrics(results)
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
                    'recommendation': result['recommendation']
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
            'multilabel_features': True
        }
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        NEW: Analyze DataFrame with pure multi-label classification
        
        Returns DataFrame with new column structure:
        - primary_aspect (replaces 'aspect')
        - secondary_aspects (new)
        - classification_type (new)
        - etc.
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        print(f"🔄 Analyzing {len(df)} reviews with pure multi-label classification...")
        
        texts = df[text_column].astype(str).tolist()
        
        # Analyze with business intelligence
        batch_results = self.analyze_batch_with_business_intelligence(texts)
        results = batch_results['individual_results']
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        output_df = df.copy()
        
        # Add new multi-label columns (NO backward compatibility)
        new_columns = [
            'sentiment', 'sentiment_confidence',
            'primary_aspect', 'secondary_aspects', 'classification_type', 
            'confidence', 'priority_level', 'severity_level',
            'business_summary', 'recommendation', 'requires_immediate_action',
            'user_experience_priority', 'mixed_concerns', 'high_priority', 'critical_severity'
        ]
        
        for col in new_columns:
            if col in results_df.columns:
                output_df[f'predicted_{col}'] = results_df[col]
        
        print(f"✅ Added {len([col for col in new_columns if col in results_df.columns])} new multi-label columns")
        
        # Add business intelligence as metadata
        output_df.attrs['business_intelligence'] = batch_results['business_intelligence']
        output_df.attrs['summary_metrics'] = batch_results['summary_metrics']
        
        return output_df
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the pipeline configuration"""
        return {
            'pipeline_type': 'Pure Multi-Label',
            'backward_compatible': False,
            'features': [
                'Multi-label aspect classification',
                'User experience prioritization',
                'Mixed concerns detection',
                'Business priority levels',
                'Severity assessment',
                'Actionable recommendations',
                'Business intelligence reporting'
            ],
            'output_format': 'New multi-label format with primary_aspect + secondary_aspects',
            'suitable_for': 'Advanced data science projects and presentations'
        }

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing Pure Multi-Label Integrated Pipeline")
    print("="*60)
    
    # Initialize pipeline
    pipeline = IntegratedMLPipeline()
    
    # Test with your FedEx sample
    test_texts = [
        "not receiving email for sign in, this app continues to be trash!",
        "Love the tracking accuracy but the interface is confusing",
        "App crashes constantly when trying to track packages",
        "Great delivery notifications but app design is ugly and hard to navigate"
    ]
    
    print("\n🧪 Testing Individual Text Analysis:")
    print("-" * 50)
    
    for i, text in enumerate(test_texts, 1):
        result = pipeline.analyze_text(text)
        
        print(f"\n{i}. Text: {text}")
        print(f"   Sentiment: {result['sentiment']} ({result['sentiment_confidence']:.3f})")
        print(f"   Primary Aspect: {result['primary_aspect']}")
        print(f"   Secondary Aspects: {result['secondary_aspects']}")
        print(f"   Classification Type: {result['classification_type']}")
        print(f"   Priority: {result['priority_level']}")
        print(f"   Severity: {result['severity_level']}")
        print(f"   Immediate Action: {result['requires_immediate_action']}")
        print(f"   Business Summary: {result['business_summary']}")
    
    print(f"\n📊 Testing Batch Analysis with Business Intelligence:")
    print("-" * 50)
    
    # Test batch analysis
    batch_results = pipeline.analyze_batch_with_business_intelligence(test_texts)
    bi = batch_results['business_intelligence']
    
    print(f"\nBusiness Intelligence Report:")
    print(f"   Total Reviews: {bi['total_reviews']}")
    print(f"   Mixed Concerns: {bi['business_metrics']['mixed_concerns_percentage']}%")
    print(f"   UX Priority: {bi['business_metrics']['user_experience_priority_percentage']}%")
    print(f"   High Priority: {bi['business_metrics']['high_priority_percentage']}%")
    print(f"   Critical Severity: {bi['business_metrics']['critical_severity_percentage']}%")
    print(f"   Immediate Action: {bi['business_metrics']['immediate_action_percentage']}%")
    
    print(f"\nTop Recommendations:")
    for rec in bi['top_recommendations']:
        print(f"   • {rec}")
    
    print(f"\nCritical Issues:")
    for i, issue in enumerate(bi['critical_issues'], 1):
        print(f"   {i}. {issue['text']} (Priority: {issue['priority_level']}, Severity: {issue['severity_level']})")
    
    # Show pipeline info
    print(f"\n⚙️ Pipeline Information:")
    info = pipeline.get_pipeline_info()
    for key, value in info.items():
        if isinstance(value, list):
            print(f"   {key.replace('_', ' ').title()}:")
            for item in value:
                print(f"      • {item}")
        else:
            print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n✅ Pure Multi-Label Pipeline Ready!")
    print(f"🎯 Perfect for your bootcamp presentation")
    print(f"🚀 Shows advanced ML concepts and business intelligence")