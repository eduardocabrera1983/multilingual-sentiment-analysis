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
    """Integrated ML Pipeline for production use"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_classifiers()
    
    def _initialize_classifiers(self):
        """Initialize classifiers"""
        try:
            if EnhancedSentimentClassifier:
                self.sentiment_classifier = EnhancedSentimentClassifier()
            else:
                self._initialize_fallback_sentiment()
            
            if EnhancedAspectClassifier:
                self.aspect_classifier = EnhancedAspectClassifier()
            else:
                self._initialize_fallback_aspect()
            
            self.logger.info("✅ Pipeline initialized")
            
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
            'product_quality': ['quality', 'performance', 'build', 'durable', 'reliable', 'fast', 'crash', 'bug'],
            'user_experience': ['easy', 'difficult', 'interface', 'design', 'intuitive', 'confusing', 'simple']
        }
        self.logger.info("✅ Fallback aspect classifier loaded")
    
    def _initialize_fallback_models(self):
        """Initialize minimal fallback models"""
        self._initialize_fallback_sentiment()
        self._initialize_fallback_aspect()
    
    def analyze_text(self, text: str, language: str = 'auto') -> Dict:
        """Analyze a single text"""
        start_time = time.time()
        
        # Sentiment analysis
        if hasattr(self, 'sentiment_classifier'):
            sentiment_result = self.sentiment_classifier.analyze_sentiment(text, language)
        else:
            sentiment_result = self._fallback_sentiment_analysis(text)
        
        # Aspect analysis
        if hasattr(self, 'aspect_classifier'):
            aspect_result = self.aspect_classifier.classify_aspect(text, language)
        else:
            aspect_result = self._fallback_aspect_analysis(text)
        
        return {
            'text': text,
            'language': language,
            'sentiment': sentiment_result['sentiment'],
            'sentiment_confidence': sentiment_result['confidence'],
            'aspect': aspect_result['aspect'],
            'aspect_confidence': aspect_result['confidence'],
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
    
    def _fallback_aspect_analysis(self, text: str) -> Dict:
        """Fallback aspect analysis"""
        text_lower = text.lower()
        scores = {'product_quality': 0, 'user_experience': 0}
        
        for aspect, keywords in self.aspect_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[aspect] += 1
        
        total = sum(scores.values())
        if total == 0:
            return {'aspect': 'general', 'confidence': 0.5, 'scores': {'product_quality': 0.0, 'user_experience': 0.0, 'general': 1.0}}
        
        normalized_scores = {k: v/total for k, v in scores.items()}
        max_aspect = max(scores, key=scores.get)
        
        return {'aspect': max_aspect, 'confidence': normalized_scores[max_aspect], 'scores': normalized_scores}
    
    def analyze_batch(self, texts: List[str], languages: List[str] = None) -> List[Dict]:
        """Analyze multiple texts"""
        if languages is None:
            languages = ['auto'] * len(texts)
        
        results = []
        for text, lang in zip(texts, languages):
            result = self.analyze_text(text, lang)
            results.append(result)
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Analyze DataFrame"""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found")
        
        texts = df[text_column].astype(str).tolist()
        results = self.analyze_batch(texts)
        
        # Add results to DataFrame
        results_df = pd.DataFrame(results)
        output_df = df.copy()
        
        for col in ['sentiment', 'sentiment_confidence', 'aspect', 'aspect_confidence']:
            if col in results_df.columns:
                output_df[f'predicted_{col}'] = results_df[col]
        
        return output_df
