#!/usr/bin/env python3
"""
Pure Multi-Label Aspect Classifier - FIXED VERSION
Save as: src/models/enhanced_aspect_classifier.py

No backward compatibility - only advanced multi-label classification
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
    print("⚠️ Transformers not available - using keyword-based classification only")

class EnhancedAspectClassifier:
    """
    Pure Multi-Label Aspect Classifier for FedEx Reviews
    
    Features:
    - Multi-label classification with primary + secondary aspects
    - User experience prioritization
    - Business priority levels (HIGH/MEDIUM/LOW)
    - Actionable recommendations
    - Mixed concerns detection
    """
    
    def __init__(self, confidence_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Priority weights (user experience first as requested)
        self.priority_weights = {
            'user_experience': 1.5,     # Highest priority
            'performance': 1.3,         # App crashes, speed issues
            'tracking_accuracy': 1.2,   # Core functionality
            'delivery_issues': 1.1,     # Business critical
            'interface_design': 1.0,    # Standard priority
            'general_satisfaction': 0.8  # Lower priority
        }
        
        # Enhanced keyword dictionary with severity weights
        self._initialize_aspect_keywords()
        
        # Initialize semantic classifier if available
        self._initialize_semantic_classifier()
        
        print("✅ Pure Multi-Label Aspect Classifier initialized")
    
    def _initialize_aspect_keywords(self):
        """Initialize comprehensive keyword system for FedEx logistics"""
        
        self.aspect_keywords = {
            'user_experience': {
                'high_severity': [
                    'impossible to use', 'cant use', 'unusable', 'terrible interface', 
                    'worst app ever', 'hate this app', 'completely unusable'
                ],
                'medium_severity': [
                    'difficult', 'confusing', 'hard to', 'complicated', 'unintuitive', 
                    'frustrating', 'annoying', 'clunky', 'awkward'
                ],
                'positive': [
                    'easy', 'simple', 'intuitive', 'user-friendly', 'straightforward', 
                    'smooth', 'elegant', 'beautiful', 'clean design'
                ],
                'keywords': [
                    'interface', 'navigation', 'menu', 'design', 'layout', 'usability', 
                    'experience', 'use', 'user', 'navigate', 'find', 'button', 'screen'
                ]
            },
            
            'performance': {
                'high_severity': [
                    'crashes', 'freezes', 'wont load', 'broken', 'stops working', 
                    'not working', 'completely broken', 'always crashes'
                ],
                'medium_severity': [
                    'slow', 'laggy', 'buggy', 'glitchy', 'hangs', 'loading forever', 
                    'takes forever', 'unresponsive'
                ],
                'positive': [
                    'fast', 'smooth', 'responsive', 'stable', 'reliable', 'quick', 
                    'works perfectly', 'no issues'
                ],
                'keywords': [
                    'performance', 'speed', 'crash', 'bug', 'error', 'loading', 
                    'response', 'freeze', 'lag', 'work', 'function'
                ]
            },
            
            'tracking_accuracy': {
                'high_severity': [
                    'wrong location', 'never updates', 'completely inaccurate', 
                    'tracking broken', 'shows wrong info', 'totally wrong'
                ],
                'medium_severity': [
                    'delayed updates', 'sometimes wrong', 'not precise', 'outdated info', 
                    'inconsistent', 'not accurate'
                ],
                'positive': [
                    'accurate', 'real-time', 'precise', 'up-to-date', 'correct', 
                    'reliable tracking', 'always accurate'
                ],
                'keywords': [
                    'tracking', 'location', 'status', 'updates', 'progress', 'package', 
                    'shipment', 'track', 'trace', 'position', 'whereabouts'
                ]
            },
            
            'delivery_issues': {
                'high_severity': [
                    'never delivered', 'lost package', 'damaged', 'wrong address', 
                    'package missing', 'never arrived'
                ],
                'medium_severity': [
                    'late delivery', 'delivery problems', 'delayed', 'missed delivery', 
                    'delivery issues', 'wrong time'
                ],
                'positive': [
                    'on time', 'perfect delivery', 'safe arrival', 'fast delivery', 
                    'delivered perfectly', 'great service'
                ],
                'keywords': [
                    'delivery', 'arrive', 'shipping', 'package', 'courier', 'driver', 
                    'deliver', 'pickup', 'drop off', 'received'
                ]
            },
            
            'interface_design': {
                'high_severity': [
                    'ugly', 'terrible design', 'horrible layout', 'looks terrible', 
                    'awful design', 'hideous interface'
                ],
                'medium_severity': [
                    'cluttered', 'messy', 'poor design', 'bad layout', 'outdated', 
                    'confusing layout', 'unclear design'
                ],
                'positive': [
                    'beautiful', 'clean', 'modern', 'attractive', 'well-designed', 
                    'sleek', 'professional', 'nice looking'
                ],
                'keywords': [
                    'design', 'look', 'appearance', 'visual', 'color', 'layout', 
                    'style', 'theme', 'graphics', 'aesthetics'
                ]
            },
            
            'general_satisfaction': {
                'high_severity': [
                    'hate this app', 'worst experience', 'never again', 'terrible app', 
                    'completely disappointed', 'total disaster'
                ],
                'medium_severity': [
                    'disappointed', 'expected better', 'not satisfied', 'mediocre', 
                    'could be better', 'not impressed'
                ],
                'positive': [
                    'love it', 'excellent', 'perfect', 'amazing', 'outstanding', 
                    'great app', 'highly recommend', 'fantastic'
                ],
                'keywords': [
                    'overall', 'general', 'experience', 'satisfied', 'recommend', 
                    'rating', 'app', 'service', 'company'
                ]
            }
        }
    
    def _initialize_semantic_classifier(self):
        """Initialize transformer-based semantic classifier"""
        if not TRANSFORMERS_AVAILABLE:
            self.semantic_classifier = None
            return
            
        try:
            # Use zero-shot classification for better aspect detection
            self.semantic_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            print("✅ Semantic classifier loaded")
        except Exception as e:
            print(f"⚠️ Could not load semantic classifier: {e}")
            self.semantic_classifier = None
    
    def classify_aspects_multilabel(self, text: str, language: str = 'en') -> Dict:
        """
        MAIN METHOD: Multi-label aspect classification
        
        Returns complete multi-label analysis with business intelligence
        """
        if not text.strip():
            return self._empty_result()
        
        text_lower = text.lower()
        
        # Step 1: Keyword-based scoring with severity weights
        keyword_scores = self._calculate_keyword_scores(text_lower)
        
        # Step 2: Semantic similarity (if available)
        semantic_scores = self._calculate_semantic_scores(text, keyword_scores.keys())
        
        # Step 3: Combine scores
        combined_scores = self._combine_scores(keyword_scores, semantic_scores)
        
        # Step 4: Apply priority weights (user experience first)
        prioritized_scores = self._apply_priority_weights(combined_scores)
        
        # Step 5: Determine multi-label classification
        result = self._determine_multilabel_classification(prioritized_scores, text)
        
        return result
    
    def _calculate_keyword_scores(self, text: str) -> Dict:
        """Calculate scores based on keyword matching with severity weighting"""
        scores = {}
        
        for aspect, keyword_dict in self.aspect_keywords.items():
            aspect_score = 0.0
            severity_multiplier = 1.0
            
            # Check for high severity indicators (weight: 3.0)
            for keyword in keyword_dict.get('high_severity', []):
                if keyword in text:
                    aspect_score += 3.0
                    severity_multiplier = 2.0
            
            # Check for medium severity indicators (weight: 2.0)  
            for keyword in keyword_dict.get('medium_severity', []):
                if keyword in text:
                    aspect_score += 2.0
                    severity_multiplier = 1.5
            
            # Check for positive indicators (weight: 1.5)
            positive_found = False
            for keyword in keyword_dict.get('positive', []):
                if keyword in text:
                    aspect_score += 1.5
                    positive_found = True
            
            # Check for general keywords (weight: 1.0)
            keyword_count = 0
            for keyword in keyword_dict.get('keywords', []):
                if keyword in text:
                    keyword_count += 1
            
            # Calculate final score
            base_score = keyword_count + aspect_score
            final_score = base_score * severity_multiplier
            
            scores[aspect] = max(final_score, 0.0)
        
        return scores
    
    def _calculate_semantic_scores(self, text: str, candidate_aspects: List) -> Dict:
        """Calculate semantic similarity scores using transformer"""
        if not self.semantic_classifier or not candidate_aspects:
            return {aspect: 0.0 for aspect in candidate_aspects}
        
        try:
            # Define semantic labels for zero-shot classification
            semantic_labels = [
                "user interface and navigation problems",
                "app performance and technical issues",
                "package tracking accuracy problems", 
                "delivery and shipping issues",
                "app design and visual appearance",
                "general satisfaction and experience"
            ]
            
            result = self.semantic_classifier(text, semantic_labels)
            
            # Map back to our aspect categories
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
                    semantic_scores[aspect] = score
            
            return semantic_scores
            
        except Exception as e:
            self.logger.warning(f"Semantic classification failed: {e}")
            return {aspect: 0.0 for aspect in candidate_aspects}
    
    def _combine_scores(self, keyword_scores: Dict, semantic_scores: Dict) -> Dict:
        """Combine keyword and semantic scores"""
        combined = {}
        all_aspects = set(keyword_scores.keys()) | set(semantic_scores.keys())
        
        for aspect in all_aspects:
            keyword_score = keyword_scores.get(aspect, 0.0)
            semantic_score = semantic_scores.get(aspect, 0.0)
            
            # Weight combination: 70% keywords, 30% semantic
            combined[aspect] = (keyword_score * 0.7) + (semantic_score * 3.0 * 0.3)
        
        return combined
    
    def _apply_priority_weights(self, scores: Dict) -> Dict:
        """Apply business priority weights (user experience first)"""
        prioritized = {}
        
        for aspect, score in scores.items():
            priority_weight = self.priority_weights.get(aspect, 1.0)
            prioritized[aspect] = score * priority_weight
        
        return prioritized
    
    def _determine_multilabel_classification(self, scores: Dict, original_text: str) -> Dict:
        """Determine multi-label classification with mixed concerns handling"""
        
        # Sort by score
        sorted_aspects = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter aspects above threshold
        significant_aspects = [(aspect, score) for aspect, score in sorted_aspects 
                             if score >= self.confidence_threshold]
        
        if not significant_aspects:
            return self._empty_result()
        
        # Determine classification type
        if len(significant_aspects) == 1:
            classification_type = "single_aspect"
            primary_aspect = significant_aspects[0][0]
            secondary_aspects = []
            
        elif len(significant_aspects) == 2:
            classification_type = "dual_aspect"
            primary_aspect = significant_aspects[0][0]
            secondary_aspects = [significant_aspects[1][0]]
            
        else:
            classification_type = "mixed_concerns"
            primary_aspect = significant_aspects[0][0]
            secondary_aspects = [aspect for aspect, _ in significant_aspects[1:3]]
        
        # Calculate confidence
        primary_confidence = significant_aspects[0][1]
        
        # Generate business summary and priority
        aspect_summary = self._create_aspect_summary(primary_aspect, secondary_aspects, classification_type)
        priority_level = self._calculate_priority_level(primary_aspect, primary_confidence, classification_type)
        
        return {
            'primary_aspect': primary_aspect,
            'secondary_aspects': secondary_aspects,
            'classification_type': classification_type,
            'confidence': primary_confidence,
            'all_scores': dict(significant_aspects),
            'priority_level': priority_level,
            'business_summary': aspect_summary,
            'review_text': original_text,
            'recommendation': self._generate_recommendation(primary_aspect, secondary_aspects, priority_level),
            'severity_level': self._calculate_severity_level(original_text, primary_aspect),
            'requires_immediate_action': priority_level == 'HIGH' and classification_type in ['dual_aspect', 'mixed_concerns']
        }
    
    def _create_aspect_summary(self, primary: str, secondary: List, type_: str) -> str:
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
        
        if type_ == "single_aspect":
            return f"Focused feedback on {primary_name}"
        elif type_ == "dual_aspect":
            secondary_name = aspect_names.get(secondary[0], secondary[0].replace('_', ' ').title())
            return f"Combined concerns: {primary_name} + {secondary_name}"
        else:
            return f"Multiple issues with {primary_name} as primary concern"
    
    def _calculate_priority_level(self, primary_aspect: str, confidence: float, type_: str) -> str:
        """Calculate business priority level"""
        base_priority = {
            'user_experience': 'HIGH',      # Your #1 priority
            'performance': 'HIGH',
            'tracking_accuracy': 'MEDIUM',
            'delivery_issues': 'MEDIUM', 
            'interface_design': 'MEDIUM',
            'general_satisfaction': 'LOW'
        }
        
        priority = base_priority.get(primary_aspect, 'MEDIUM')
        
        # Boost priority for mixed concerns or high confidence
        if type_ == "mixed_concerns" or confidence > 2.0:
            if priority == 'MEDIUM':
                priority = 'HIGH'
            elif priority == 'LOW':
                priority = 'MEDIUM'
        
        return priority
    
    def _calculate_severity_level(self, text: str, primary_aspect: str) -> str:
        """Calculate severity based on language intensity"""
        text_lower = text.lower()
        
        # Check for severe language indicators
        severe_indicators = [
            'terrible', 'horrible', 'worst', 'hate', 'broken', 'useless', 
            'trash', 'garbage', 'awful', 'disaster', 'completely', 'totally'
        ]
        
        severe_count = sum(1 for indicator in severe_indicators if indicator in text_lower)
        
        if severe_count >= 2:
            return 'CRITICAL'
        elif severe_count == 1:
            return 'HIGH'
        else:
            return 'MODERATE'
    
    def _generate_recommendation(self, primary: str, secondary: List, priority: str) -> str:
        """Generate actionable recommendations"""
        recommendations = {
            'user_experience': "IMMEDIATE: Route to UX team for interface redesign",
            'performance': "URGENT: Escalate to engineering team for technical fixes",
            'tracking_accuracy': "Review with logistics and API integration teams",
            'delivery_issues': "Forward to operations and delivery management",
            'interface_design': "Share with design team for visual improvements",
            'general_satisfaction': "Monitor for trends and overall satisfaction metrics"
        }
        
        base_rec = recommendations.get(primary, "Review and categorize for appropriate team")
        
        if secondary:
            secondary_teams = {
                'user_experience': 'UX',
                'performance': 'Engineering', 
                'tracking_accuracy': 'Logistics',
                'delivery_issues': 'Operations',
                'interface_design': 'Design',
                'general_satisfaction': 'Customer Success'
            }
            
            secondary_team_names = [secondary_teams.get(aspect, 'Review') for aspect in secondary]
            return f"{base_rec}. Also coordinate with: {', '.join(secondary_team_names)} teams"
        else:
            return base_rec
    
    def _empty_result(self) -> Dict:
        """Return empty classification result"""
        return {
            'primary_aspect': 'general_satisfaction',
            'secondary_aspects': [],
            'classification_type': 'unclear',
            'confidence': 0.0,
            'all_scores': {},
            'priority_level': 'LOW',
            'business_summary': 'No clear aspect detected',
            'review_text': '',
            'recommendation': 'Review manually for proper categorization',
            'severity_level': 'MODERATE',
            'requires_immediate_action': False
        }
    
    def analyze_batch(self, texts: List[str], languages: List[str] = None) -> List[Dict]:
        """Analyze batch of texts with multi-label classification"""
        if languages is None:
            languages = ['en'] * len(texts)
        
        results = []
        for i, text in enumerate(texts):
            lang = languages[i] if i < len(languages) else 'en'
            result = self.classify_aspects_multilabel(text, lang)
            results.append(result)
        
        return results
    
    def generate_business_report(self, results: List[Dict]) -> Dict:
        """Generate comprehensive business intelligence report"""
        
        total_reviews = len(results)
        if total_reviews == 0:
            return {}
        
        # Count classifications by type
        classification_types = Counter([r['classification_type'] for r in results])
        
        # Count primary aspects
        primary_aspects = Counter([r['primary_aspect'] for r in results])
        
        # Count priority levels
        priority_levels = Counter([r['priority_level'] for r in results])
        
        # Count severity levels
        severity_levels = Counter([r['severity_level'] for r in results])
        
        # Find mixed concerns patterns
        mixed_concerns = [r for r in results if r['classification_type'] == 'mixed_concerns']
        mixed_patterns = Counter()
        for review in mixed_concerns:
            pattern = f"{review['primary_aspect']} + {', '.join(review['secondary_aspects'])}"
            mixed_patterns[pattern] += 1
        
        # Calculate business metrics
        ux_related = [r for r in results if 'user_experience' in ([r['primary_aspect']] + r['secondary_aspects'])]
        ux_priority_percentage = (len(ux_related) / total_reviews) * 100 if total_reviews > 0 else 0
        
        immediate_action_needed = sum(1 for r in results if r['requires_immediate_action'])
        
        return {
            'summary': {
                'total_reviews_analyzed': total_reviews,
                'user_experience_priority_percentage': round(ux_priority_percentage, 1),
                'mixed_concerns_percentage': round((classification_types['mixed_concerns'] / total_reviews) * 100, 1) if total_reviews > 0 else 0,
                'immediate_action_required': immediate_action_needed,
                'immediate_action_percentage': round((immediate_action_needed / total_reviews) * 100, 1) if total_reviews > 0 else 0
            },
            'classification_breakdown': {
                'by_type': dict(classification_types),
                'by_primary_aspect': dict(primary_aspects),
                'by_priority_level': dict(priority_levels),
                'by_severity_level': dict(severity_levels)
            },
            'mixed_concerns_patterns': dict(mixed_patterns.most_common(5)),
            'top_recommendations': self._generate_top_recommendations(results),
            'user_experience_insights': self._generate_ux_insights(results),
            'critical_issues': self._identify_critical_issues(results)
        }
    
    def _generate_top_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate top actionable recommendations"""
        high_priority = [r for r in results if r['priority_level'] == 'HIGH']
        critical_severity = [r for r in results if r['severity_level'] == 'CRITICAL']
        
        recommendations = []
        
        if len(critical_severity) > 0:
            recommendations.append(f"CRITICAL: {len(critical_severity)} reviews require immediate attention")
        
        if len(high_priority) > len(results) * 0.3:
            recommendations.append(f"URGENT: {len(high_priority)} high-priority issues detected")
        
        # Most common high-priority issues
        if high_priority:
            common_issues = Counter([r['primary_aspect'] for r in high_priority]).most_common(2)
            for issue, count in common_issues:
                recommendations.append(f"Focus on {issue.replace('_', ' ')}: {count} high-priority reports")
        
        return recommendations[:5]
    
    def _generate_ux_insights(self, results: List[Dict]) -> Dict:
        """Generate specific UX insights (user experience prioritization)"""
        ux_reviews = [r for r in results if r['primary_aspect'] == 'user_experience']
        ux_mixed = [r for r in results if 'user_experience' in r['secondary_aspects']]
        
        return {
            'dedicated_ux_issues': len(ux_reviews),
            'ux_in_mixed_concerns': len(ux_mixed),
            'total_ux_mentions': len(ux_reviews) + len(ux_mixed),
            'ux_priority_score': round(sum([r['confidence'] for r in ux_reviews]) / max(len(ux_reviews), 1), 2),
            'ux_severity_distribution': dict(Counter([r['severity_level'] for r in ux_reviews]))
        }
    
    def _identify_critical_issues(self, results: List[Dict]) -> List[Dict]:
        """Identify reviews requiring immediate action"""
        critical_reviews = [
            r for r in results 
            if r['requires_immediate_action'] or r['severity_level'] == 'CRITICAL'
        ]
        
        return sorted(critical_reviews, key=lambda x: x['confidence'], reverse=True)[:5]

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing Pure Multi-Label Aspect Classifier")
    print("="*60)
    
    # Initialize classifier
    classifier = EnhancedAspectClassifier()
    
    # Test texts for your presentation
    test_texts = [
        # Your actual sample from FedEx data
        "not receiving email for sign in, this app continues to be trash!",
        
        # Mixed concerns examples
        "Love the tracking accuracy but the interface is confusing",
        "App crashes frequently and the interface is terrible", 
        "Great delivery notifications but app design is ugly and hard to navigate",
        
        # User experience priority
        "Interface is impossible to use, terrible navigation and confusing layout",
        
        # Performance issues
        "App crashes constantly when trying to track packages"
    ]
    
    print("\n🧪 Testing Multi-Label Classifications:")
    print("-" * 50)
    
    results = []
    for i, text in enumerate(test_texts, 1):
        result = classifier.classify_aspects_multilabel(text)
        results.append(result)
        
        print(f"\n{i}. Text: {text}")
        print(f"   Primary: {result['primary_aspect']}")
        print(f"   Secondary: {result['secondary_aspects']}")
        print(f"   Type: {result['classification_type']}")
        print(f"   Priority: {result['priority_level']}")
        print(f"   Severity: {result['severity_level']}")
        print(f"   Immediate Action: {result['requires_immediate_action']}")
        print(f"   Business: {result['business_summary']}")
        print(f"   Recommendation: {result['recommendation']}")
    
    # Generate business report
    print(f"\n📊 Business Intelligence Report:")
    print("="*60)
    
    report = classifier.generate_business_report(results)
    
    print(f"\n📈 Summary:")
    for key, value in report['summary'].items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n🎯 Classification Breakdown:")
    print(f"   By Type: {report['classification_breakdown']['by_type']}")
    print(f"   By Priority: {report['classification_breakdown']['by_priority_level']}")
    print(f"   By Severity: {report['classification_breakdown']['by_severity_level']}")
    
    print(f"\n🔀 Mixed Concerns Patterns:")
    for pattern, count in report['mixed_concerns_patterns'].items():
        print(f"   {pattern}: {count} times")
    
    print(f"\n💡 Top Recommendations:")
    for rec in report['top_recommendations']:
        print(f"   • {rec}")
    
    print(f"\n🎨 User Experience Insights:")
    ux_insights = report['user_experience_insights']
    for key, value in ux_insights.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n🚨 Critical Issues Requiring Action:")
    for i, issue in enumerate(report['critical_issues'], 1):
        print(f"   {i}. {issue['review_text'][:60]}... (Priority: {issue['priority_level']}, Severity: {issue['severity_level']})")
    
    print(f"\n✅ Pure Multi-Label Classification System Ready!")
    print(f"🎯 Perfect for bootcamp presentation - shows advanced ML concepts")