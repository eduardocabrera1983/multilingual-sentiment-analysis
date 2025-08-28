#!/usr/bin/env python3
"""
Enhanced Multi-Label Aspect Classifier - UPDATED for Two-Model Ensemble Integration
Save as: src/models/enhanced_aspect_classifier.py

INTEGRATION UPDATES:
- Compatible with new two-model ensemble sentiment classifier
- Improved confidence normalization
- Better sentiment-aware classification
- Enhanced business intelligence integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
from collections import Counter, defaultdict

# Try to import transformers for semantic analysis
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available - using keyword-based classification only")

class EnhancedAspectClassifier:
    """
    Enhanced Multi-Label Aspect Classifier for FedEx Reviews
    Updated for integration with two-model ensemble sentiment classifier
    """
    
    def __init__(self, confidence_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Priority weights (user experience first)
        self.priority_weights = {
            'user_experience': 1.5,     # Highest priority
            'performance': 1.3,         # App crashes, speed issues
            'tracking_accuracy': 1.2,   # Core functionality
            'delivery_issues': 1.1,     # Business critical
            'interface_design': 1.0,    # Standard priority
            'general_satisfaction': 0.8  # Lower priority
        }
        
        # Initialize improved keyword system
        self._initialize_enhanced_keywords()
        
        # Initialize semantic classifier if available
        self._initialize_semantic_classifier()
        
        print("Enhanced Aspect Classifier V2.0 initialized (Two-Model Ensemble Compatible)")
    
    def _initialize_enhanced_keywords(self):
        """Initialize improved keyword system with context patterns"""
        
        # Context-aware patterns for better matching
        self.aspect_patterns = {
            'user_experience': {
                'strong_patterns': [
                    r'\b(impossible|difficult|hard|easy|simple)\s+to\s+(use|navigate|find)',
                    r'\b(terrible|great|excellent|awful)\s+(interface|experience|ux)',
                    r'\b(confusing|intuitive|clear)\s+(navigation|menu|layout)',
                    r'\buser[\s\-]friendly\b',
                    r'\b(love|hate)\s+(using|the\s+interface)'
                ],
                'high_severity': ['impossible to use', 'cant use', 'unusable', 'worst experience'],
                'medium_severity': ['difficult', 'confusing', 'hard to', 'complicated', 'frustrating'],
                'positive': ['easy to use', 'simple', 'intuitive', 'user-friendly', 'straightforward'],
                'keywords': {
                    'interface': 0.7, 'navigation': 0.7, 'menu': 0.6, 
                    'design': 0.5, 'layout': 0.6, 'usability': 0.8,
                    'experience': 0.6, 'use': 0.4, 'user': 0.5
                }
            },
            
            'performance': {
                'strong_patterns': [
                    r'\b(app|application)\s+(crashes|freezes|hangs)',
                    r'\b(always|constantly|keeps)\s+(crashing|freezing)',
                    r'\b(slow|fast|quick|laggy)\s+(loading|response|performance)',
                    r'\bworks\s+(perfectly|great|well|smoothly)',
                    r'\b(broken|buggy|glitchy)\s+(app|system)'
                ],
                'high_severity': ['crashes', 'freezes', 'wont load', 'broken', 'not working'],
                'medium_severity': ['slow', 'laggy', 'buggy', 'glitchy', 'unresponsive'],
                'positive': ['fast', 'smooth', 'responsive', 'stable', 'reliable', 'works perfectly'],
                'keywords': {
                    'performance': 0.9, 'speed': 0.8, 'crash': 1.0, 
                    'bug': 0.9, 'error': 0.8, 'loading': 0.7,
                    'freeze': 1.0, 'lag': 0.8, 'works': 0.3
                }
            },
            
            'tracking_accuracy': {
                'strong_patterns': [
                    r'\b(tracking|location)\s+(wrong|incorrect|accurate|precise)',
                    r'\b(never|always)\s+updates',
                    r'\b(real[\s\-]time|live)\s+(tracking|updates)',
                    r'\bshows?\s+(wrong|correct|accurate)\s+(location|status)'
                ],
                'high_severity': ['wrong location', 'never updates', 'tracking broken', 'totally wrong'],
                'medium_severity': ['delayed updates', 'not accurate', 'inconsistent tracking'],
                'positive': ['accurate tracking', 'real-time', 'precise', 'up-to-date', 'correct location'],
                'keywords': {
                    'tracking': 1.0, 'location': 0.9, 'status': 0.7, 
                    'updates': 0.6, 'package': 0.5, 'shipment': 0.6,
                    'track': 0.8, 'accurate': 0.8
                }
            },
            
            'delivery_issues': {
                'strong_patterns': [
                    r'\b(never|not)\s+delivered',
                    r'\b(lost|damaged|missing)\s+package',
                    r'\b(late|delayed|on[\s\-]time)\s+delivery',
                    r'\bdelivered?\s+to\s+(wrong|correct)\s+address'
                ],
                'high_severity': ['never delivered', 'lost package', 'damaged', 'package missing'],
                'medium_severity': ['late delivery', 'delayed', 'missed delivery'],
                'positive': ['on time', 'perfect delivery', 'fast delivery', 'delivered perfectly'],
                'keywords': {
                    'delivery': 1.0, 'arrive': 0.7, 'shipping': 0.8, 
                    'package': 0.6, 'courier': 0.7, 'deliver': 0.9
                }
            },
            
            'interface_design': {
                'strong_patterns': [
                    r'\b(ugly|beautiful|clean|messy)\s+(design|interface|layout)',
                    r'\b(modern|outdated|sleek)\s+(look|appearance|design)',
                    r'\blooks?\s+(terrible|great|amazing|awful)'
                ],
                'high_severity': ['ugly', 'terrible design', 'horrible layout', 'awful design'],
                'medium_severity': ['cluttered', 'messy', 'poor design', 'outdated look'],
                'positive': ['beautiful', 'clean design', 'modern', 'sleek', 'well-designed'],
                'keywords': {
                    'design': 0.8, 'look': 0.5, 'appearance': 0.7, 
                    'visual': 0.7, 'layout': 0.6, 'style': 0.6
                }
            },
            
            'general_satisfaction': {
                'strong_patterns': [
                    r'\b(recommend|not\s+recommend)\s+(to|it)',
                    r'\b(best|worst)\s+(app|service|experience)',
                    r'\b(love|hate)\s+(this|the)\s+app',
                    r'\b(satisfied|disappointed)\s+with'
                ],
                'high_severity': ['hate this app', 'worst experience', 'never again', 'terrible app'],
                'medium_severity': ['disappointed', 'not satisfied', 'mediocre'],
                'positive': ['love it', 'excellent app', 'highly recommend', 'very satisfied'],
                'keywords': {
                    'recommend': 0.9, 'overall': 0.5, 'satisfied': 0.8,
                    'experience': 0.4, 'app': 0.3, 'good': 0.4,
                    'excellent': 0.7, 'terrible': 0.7
                }
            }
        }
    
    def _initialize_semantic_classifier(self):
        """Initialize transformer-based semantic classifier"""
        if not TRANSFORMERS_AVAILABLE:
            self.semantic_classifier = None
            return
            
        try:
            self.semantic_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            print("Semantic classifier loaded successfully")
        except Exception as e:
            print(f"Could not load semantic classifier: {e}")
            self.semantic_classifier = None
    
    def classify_aspects_multilabel(self, text: str, language: str = 'en', 
                                   sentiment: str = 'neutral', 
                                   sentiment_confidence: float = 0.5) -> Dict:
        """
        Main classification method with proper confidence normalization
        Compatible with two-model ensemble sentiment input
        """
        if not text.strip():
            return self._empty_result()
        
        text_lower = text.lower()
        
        # Step 1: Calculate enhanced keyword scores
        keyword_scores = self._calculate_enhanced_scores(text_lower, text)
        
        # Step 2: Semantic similarity (if available)
        semantic_scores = self._calculate_semantic_scores(text, keyword_scores.keys())
        
        # Step 3: Combine and normalize scores
        combined_scores = self._combine_and_normalize_scores(
            keyword_scores, semantic_scores, len(text.split())
        )
        
        # Step 4: Apply priority weights
        prioritized_scores = self._apply_priority_weights(combined_scores)
        
        # Step 5: Determine multi-label classification with dynamic thresholds
        result = self._determine_multilabel_with_dynamic_thresholds(
            prioritized_scores, text, sentiment, sentiment_confidence
        )
        
        return result
    
    def _calculate_enhanced_scores(self, text_lower: str, original_text: str) -> Dict:
        """Calculate scores using patterns and weighted keywords"""
        scores = {}
        
        for aspect, config in self.aspect_patterns.items():
            score = 0.0
            matches = []
            
            # Check strong patterns (highest weight)
            for pattern in config.get('strong_patterns', []):
                if re.search(pattern, text_lower):
                    score += 1.5
                    matches.append('pattern')
            
            # Check severity keywords
            severity_modifier = 1.0
            for keyword in config.get('high_severity', []):
                if keyword in text_lower:
                    score += 1.2
                    severity_modifier = 1.5
                    matches.append('high_severity')
            
            for keyword in config.get('medium_severity', []):
                if keyword in text_lower:
                    score += 0.8
                    severity_modifier = max(severity_modifier, 1.2)
                    matches.append('medium_severity')
            
            # Check positive indicators
            for keyword in config.get('positive', []):
                if keyword in text_lower:
                    score += 0.7
                    matches.append('positive')
            
            # Check weighted keywords
            for keyword, weight in config.get('keywords', {}).items():
                if keyword in text_lower:
                    if self._check_keyword_context(text_lower, keyword):
                        score += weight
                        matches.append(f'keyword_{keyword}')
            
            # Apply severity modifier
            final_score = score * severity_modifier
            
            # Normalize to 0-1 range using sigmoid-like function
            normalized_score = np.tanh(final_score / 3.0)
            scores[aspect] = min(1.0, max(0.0, normalized_score))
        
        return scores
    
    def _check_keyword_context(self, text: str, keyword: str, window: int = 30) -> bool:
        """Check if keyword appears in relevant context"""
        idx = text.find(keyword)
        if idx == -1:
            return False
        
        start = max(0, idx - window)
        end = min(len(text), idx + len(keyword) + window)
        context = text[start:end]
        
        # Check for negations
        negations = ['not', 'no', "doesn't", "won't", "can't", "isn't", "never"]
        for neg in negations:
            if neg in context and abs(context.find(neg) - context.find(keyword)) < 10:
                return False
        
        return True
    
    def _calculate_semantic_scores(self, text: str, candidate_aspects: List) -> Dict:
        """Calculate semantic similarity scores"""
        if not self.semantic_classifier or not candidate_aspects:
            return {aspect: 0.0 for aspect in candidate_aspects}
        
        try:
            semantic_labels = [
                "user interface and navigation problems",
                "app performance and technical issues",
                "package tracking accuracy problems", 
                "delivery and shipping issues",
                "app design and visual appearance",
                "general satisfaction and experience"
            ]
            
            result = self.semantic_classifier(text, semantic_labels)
            
            label_mapping = {
                "user interface and navigation problems": "user_experience",
                "app performance and technical issues": "performance", 
                "package tracking accuracy problems": "tracking_accuracy",
                "delivery and shipping issues": "delivery_issues",
                "app design and visual appearance": "interface_design",
                "general satisfaction and experience": "general_satisfaction"
            }
            
            semantic_scores = {}
            for label, score in zip(result['labels'], result['scores']):
                if label in label_mapping:
                    aspect = label_mapping[label]
                    semantic_scores[aspect] = min(1.0, score)
            
            return semantic_scores
            
        except Exception as e:
            self.logger.warning(f"Semantic classification failed: {e}")
            return {aspect: 0.0 for aspect in candidate_aspects}
    
    def _combine_and_normalize_scores(self, keyword_scores: Dict, 
                                     semantic_scores: Dict, 
                                     text_length: int) -> Dict:
        """Combine scores with proper normalization"""
        combined = {}
        all_aspects = set(keyword_scores.keys()) | set(semantic_scores.keys())
        
        # Calculate length penalty
        length_factor = min(1.0, text_length / 20.0)
        
        for aspect in all_aspects:
            keyword_score = keyword_scores.get(aspect, 0.0)
            semantic_score = semantic_scores.get(aspect, 0.0)
            
            # Weighted combination
            if semantic_score > 0:
                combined_score = (keyword_score * 0.7) + (semantic_score * 0.3)
            else:
                combined_score = keyword_score
            
            # Apply length factor
            final_score = combined_score * (0.8 + 0.2 * length_factor)
            combined[aspect] = min(1.0, max(0.0, final_score))
        
        return combined
    
    def _apply_priority_weights(self, scores: Dict) -> Dict:
        """Apply business priority weights"""
        prioritized = {}
        
        for aspect, score in scores.items():
            priority_weight = self.priority_weights.get(aspect, 1.0)
            weighted_score = score * priority_weight
            prioritized[aspect] = min(1.0, weighted_score)
        
        return prioritized
    
    def _determine_multilabel_with_dynamic_thresholds(self, scores: Dict, 
                                                     original_text: str,
                                                     sentiment: str,
                                                     sentiment_confidence: float) -> Dict:
        """Determine classification with dynamic thresholds based on sentiment"""
        
        # Dynamic thresholds based on sentiment
        if sentiment == 'positive':
            primary_threshold = 0.20
            secondary_threshold = 0.15
        elif sentiment == 'negative':
            primary_threshold = 0.35
            secondary_threshold = 0.30
        else:
            primary_threshold = 0.30
            secondary_threshold = 0.25
        
        # Sort by score
        sorted_aspects = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter significant aspects
        significant_aspects = []
        for aspect, score in sorted_aspects:
            if score >= primary_threshold:
                significant_aspects.append((aspect, score))
            elif len(significant_aspects) > 0 and score >= secondary_threshold:
                if score >= significant_aspects[0][1] * 0.5:
                    significant_aspects.append((aspect, score))
        
        if not significant_aspects:
            return self._empty_result()
        
        # Determine classification type based on sentiment
        primary_aspect = significant_aspects[0][0]
        primary_confidence = min(1.0, significant_aspects[0][1])
        
        if len(significant_aspects) == 1:
            secondary_aspects = []
            if sentiment == 'positive':
                classification_type = "focused_praise"
            elif sentiment == 'negative':
                classification_type = "single_concern"
            else:
                classification_type = "single_aspect"
                
        elif len(significant_aspects) == 2:
            secondary_aspects = [significant_aspects[1][0]]
            if sentiment == 'positive':
                classification_type = "dual_strengths"
            elif sentiment == 'negative':
                classification_type = "dual_concerns"
            else:
                classification_type = "dual_aspect"
                
        else:
            secondary_aspects = [aspect for aspect, _ in significant_aspects[1:3]]
            if sentiment == 'positive':
                classification_type = "multiple_strengths"
            elif sentiment == 'negative':
                classification_type = "mixed_concerns"
            else:
                classification_type = "mixed_feedback"
        
        # Generate business intelligence
        aspect_summary = self._create_aspect_summary(
            primary_aspect, secondary_aspects, classification_type, sentiment
        )
        priority_level = self._calculate_priority_level(
            primary_aspect, primary_confidence, classification_type, sentiment, sentiment_confidence
        )
        recommendation = self._generate_recommendation(
            primary_aspect, secondary_aspects, priority_level, sentiment
        )
        severity_level = self._calculate_severity_level(
            original_text, primary_aspect, sentiment
        )
        
        return {
            'primary_aspect': primary_aspect,
            'secondary_aspects': secondary_aspects,
            'classification_type': classification_type,
            'confidence': primary_confidence,
            'confidence_percentage': f"{primary_confidence * 100:.1f}%",
            'all_scores': {k: min(1.0, v) for k, v in significant_aspects},
            'priority_level': priority_level,
            'business_summary': aspect_summary,
            'review_text': original_text,
            'recommendation': recommendation,
            'severity_level': severity_level,
            'requires_immediate_action': priority_level in ['HIGH', 'CRITICAL'] and sentiment == 'negative',
            'sentiment': sentiment,
            'sentiment_confidence': sentiment_confidence
        }
    
    def _create_aspect_summary(self, primary: str, secondary: List, 
                              type_: str, sentiment: str) -> str:
        """Create business-friendly summary"""
        aspect_names = {
            'user_experience': 'User Experience',
            'performance': 'App Performance',
            'tracking_accuracy': 'Tracking Accuracy', 
            'delivery_issues': 'Delivery Issues',
            'interface_design': 'Interface Design',
            'general_satisfaction': 'Overall Satisfaction'
        }
        
        primary_name = aspect_names.get(primary, primary.replace('_', ' ').title())
        
        if sentiment == 'positive':
            if type_ == "focused_praise":
                return f"Positive feedback on {primary_name}"
            elif type_ == "dual_strengths":
                secondary_name = aspect_names.get(secondary[0], secondary[0].replace('_', ' ').title())
                return f"Praise for {primary_name} and {secondary_name}"
            elif type_ == "multiple_strengths":
                return f"Multiple positive aspects led by {primary_name}"
        
        elif sentiment == 'negative':
            if type_ == "single_concern":
                return f"Issue reported with {primary_name}"
            elif type_ == "dual_concerns":
                secondary_name = aspect_names.get(secondary[0], secondary[0].replace('_', ' ').title())
                return f"Problems with {primary_name} and {secondary_name}"
            elif type_ == "mixed_concerns":
                return f"Multiple issues, primarily {primary_name}"
        
        else:  # neutral
            if type_ == "single_aspect":
                return f"Feedback about {primary_name}"
            elif type_ == "dual_aspect":
                secondary_name = aspect_names.get(secondary[0], secondary[0].replace('_', ' ').title())
                return f"Comments on {primary_name} and {secondary_name}"
            elif type_ == "mixed_feedback":
                return f"Mixed feedback, mainly about {primary_name}"
        
        return f"Feedback on {primary_name}"
    
    def _calculate_priority_level(self, primary_aspect: str, confidence: float,
                                 type_: str, sentiment: str, 
                                 sentiment_confidence: float) -> str:
        """Calculate business priority level"""
        
        if sentiment == 'positive':
            return 'LOW'
        
        base_priority = {
            'user_experience': 'HIGH',
            'performance': 'HIGH',
            'tracking_accuracy': 'MEDIUM',
            'delivery_issues': 'MEDIUM', 
            'interface_design': 'MEDIUM',
            'general_satisfaction': 'LOW'
        }
        
        aspect_priority = base_priority.get(primary_aspect, 'MEDIUM')
        
        if sentiment == 'negative':
            if confidence >= 0.7 and sentiment_confidence >= 0.7:
                return aspect_priority
            elif confidence >= 0.5:
                return 'MEDIUM' if aspect_priority == 'HIGH' else 'LOW'
            else:
                return 'LOW'
        
        return 'LOW'
    
    def _generate_recommendation(self, primary: str, secondary: List,
                                priority: str, sentiment: str) -> str:
        """Generate actionable recommendations"""
        
        if sentiment == 'positive':
            positive_recs = {
                'user_experience': "SUCCESS: UX working well - document best practices",
                'performance': "SUCCESS: Performance satisfactory - maintain standards",
                'tracking_accuracy': "SUCCESS: Tracking accurate - continue monitoring",
                'delivery_issues': "SUCCESS: Delivery performing well",
                'interface_design': "SUCCESS: Design appreciated - note successful elements",
                'general_satisfaction': "SUCCESS: High satisfaction - share with team"
            }
            return positive_recs.get(primary, "SUCCESS: Positive feedback received")
        
        elif sentiment == 'negative':
            negative_recs = {
                'user_experience': "ACTION: Route to UX team for improvements",
                'performance': "URGENT: Escalate to engineering team",
                'tracking_accuracy': "ACTION: Review with logistics team",
                'delivery_issues': "ACTION: Forward to operations",
                'interface_design': "ACTION: Share with design team",
                'general_satisfaction': "REVIEW: Analyze for specific issues"
            }
            
            base = negative_recs.get(primary, "ACTION: Review and assign")
            if secondary:
                return f"{base}. Also affects: {', '.join(secondary)}"
            return base
        
        return "MONITOR: Track feedback trends"
    
    def _calculate_severity_level(self, text: str, primary_aspect: str,
                                 sentiment: str) -> str:
        """Calculate severity based on language and sentiment"""
        
        if sentiment == 'positive':
            return 'LOW'
        
        text_lower = text.lower()
        
        severe_indicators = [
            'terrible', 'horrible', 'worst', 'hate', 'broken', 'useless',
            'never works', 'always crashes', 'impossible'
        ]
        
        moderate_indicators = [
            'bad', 'poor', 'disappointing', 'frustrating', 'annoying',
            'slow', 'confusing', 'problems'
        ]
        
        severe_count = sum(1 for ind in severe_indicators if ind in text_lower)
        moderate_count = sum(1 for ind in moderate_indicators if ind in text_lower)
        
        if sentiment == 'negative':
            if severe_count >= 2:
                return 'CRITICAL'
            elif severe_count >= 1:
                return 'HIGH'
            elif moderate_count >= 2:
                return 'MODERATE'
            else:
                return 'MODERATE'
        
        return 'LOW'
    
    def _empty_result(self) -> Dict:
        """Return empty classification result"""
        return {
            'primary_aspect': 'general_satisfaction',
            'secondary_aspects': [],
            'classification_type': 'unclear',
            'confidence': 0.0,
            'confidence_percentage': '0.0%',
            'all_scores': {},
            'priority_level': 'LOW',
            'business_summary': 'No clear aspect detected',
            'review_text': '',
            'recommendation': 'Review manually for proper categorization',
            'severity_level': 'LOW',
            'requires_immediate_action': False,
            'sentiment': 'neutral',
            'sentiment_confidence': 0.0
        }
    
    def analyze_batch(self, texts: List[str], sentiments: List[Dict] = None) -> List[Dict]:
        """Analyze batch of texts with sentiment context"""
        if sentiments is None:
            sentiments = [{'sentiment': 'neutral', 'confidence': 0.5}] * len(texts)
        
        results = []
        for text, sentiment_data in zip(texts, sentiments):
            result = self.classify_aspects_multilabel(
                text=text,
                sentiment=sentiment_data.get('sentiment', 'neutral'),
                sentiment_confidence=sentiment_data.get('confidence', 0.5)
            )
            results.append(result)
        
        return results

    def generate_business_report(self, results: List[Dict]) -> Dict:
        """Generate comprehensive business intelligence report"""
        if not results:
            return {'summary': {}, 'top_recommendations': [], 'critical_issues': []}
        
        total = len(results)
        
        # Aggregate metrics
        priority_counts = Counter([r.get('priority_level', 'MEDIUM') for r in results])
        severity_counts = Counter([r.get('severity_level', 'MODERATE') for r in results])
        aspect_counts = Counter([r.get('primary_aspect', 'general_satisfaction') for r in results])
        
        # Calculate percentages
        high_priority_pct = (priority_counts.get('HIGH', 0) / total) * 100
        critical_severity_pct = (severity_counts.get('CRITICAL', 0) / total) * 100
        immediate_action_count = sum(1 for r in results if r.get('requires_immediate_action', False))
        
        return {
            'summary': {
                'total_reviews': total,
                'high_priority_percentage': round(high_priority_pct, 1),
                'critical_severity_percentage': round(critical_severity_pct, 1),
                'immediate_action_required': immediate_action_count
            },
            'top_recommendations': self._generate_top_recommendations(results),
            'aspect_distribution': dict(aspect_counts.most_common(5)),
            'priority_distribution': dict(priority_counts),
            'severity_distribution': dict(severity_counts)
        }
    
    def _generate_top_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate actionable business recommendations"""
        recommendations = []
        
        critical_count = sum(1 for r in results if r.get('severity_level') == 'CRITICAL')
        if critical_count > 0:
            recommendations.append(f"CRITICAL: {critical_count} reviews need immediate attention")
        
        high_priority = sum(1 for r in results if r.get('priority_level') == 'HIGH')
        if high_priority > len(results) * 0.2:
            recommendations.append(f"HIGH VOLUME: {high_priority} high-priority issues")
        
        # Most common aspects
        aspects = Counter([r.get('primary_aspect') for r in results])
        top_aspect = aspects.most_common(1)[0] if aspects else ('general_satisfaction', 0)
        recommendations.append(f"Focus on {top_aspect[0].replace('_', ' ')}: {top_aspect[1]} mentions")
        
        return recommendations[:5]


# Test the updated classifier
if __name__ == "__main__":
    print("Testing Enhanced Aspect Classifier V2.0 (Two-Model Ensemble Compatible)")
    print("="*70)
    
    classifier = EnhancedAspectClassifier()
    
    test_cases = [
        {
            'text': "The app works so good I want to recommend it to all my colleagues.",
            'sentiment': 'positive',
            'sentiment_confidence': 0.95
        },
        {
            'text': "App crashes constantly and interface is terrible, deliveries are late",
            'sentiment': 'negative',
            'sentiment_confidence': 0.90
        },
        {
            'text': "The tracking works but interface could be better",
            'sentiment': 'neutral',
            'sentiment_confidence': 0.60
        },
        {
            'text': "not receiving email for sign in, this app continues to be trash!",
            'sentiment': 'negative',
            'sentiment_confidence': 0.85
        }
    ]
    
    print("\nClassification Results:")
    print("-" * 70)
    
    for i, test in enumerate(test_cases, 1):
        result = classifier.classify_aspects_multilabel(
            text=test['text'],
            sentiment=test['sentiment'],
            sentiment_confidence=test['sentiment_confidence']
        )
        
        print(f"\n{i}. Text: {test['text'][:60]}...")
        print(f"   Sentiment: {test['sentiment']} ({test['sentiment_confidence']*100:.0f}%)")
        print(f"   Classification: {result['classification_type']}")
        print(f"   Primary: {result['primary_aspect']}")
        print(f"   Secondary: {result['secondary_aspects']}")
        print(f"   Confidence: {result['confidence_percentage']}")
        print(f"   Priority: {result['priority_level']}")
        print(f"   Summary: {result['business_summary']}")
        print(f"   Recommendation: {result['recommendation']}")
    
    print("\nEnhanced Aspect Classifier V2.0 ready for two-model ensemble integration!")