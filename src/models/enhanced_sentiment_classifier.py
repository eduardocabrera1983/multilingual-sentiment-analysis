#!/usr/bin/env python3
"""
Enhanced GPU-Accelerated Sentiment Classifier - VERSION 2.0
Save as: src/models/enhanced_sentiment_classifier.py

IMPROVEMENTS IN V2.0:
1. Better error handling and fallback mechanisms
2. Proper confidence normalization
3. Improved GPU memory management
4. Better integration with aspect classifier
5. More robust model loading
"""

import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
from typing import Dict, List, Optional, Tuple
import time
import logging
from datetime import datetime
import os
import gc

warnings.filterwarnings('ignore')

class EnhancedSentimentClassifier:
    """
    Enhanced GPU-Accelerated Sentiment Classifier V2.0
    Optimized for NVIDIA RTX 4000 Ada (12GB VRAM)
    """
    
    def __init__(self, use_ensemble=False, device='auto', verbose=True):
        self.use_ensemble = use_ensemble
        self.verbose = verbose
        self.models = {}
        self.device = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Configure device
        self.device = self._configure_device(device)
        
        # Print initialization info
        self._print_init_info()
        
        # Load models with improved error handling
        self._load_models_improved()
        
        # Initialize cache for better performance
        self.cache = {}
        self.max_cache_size = 1000
    
    def _configure_device(self, device_preference: str) -> str:
        """Configure device with automatic fallback"""
        if device_preference == 'auto':
            if torch.cuda.is_available():
                # Clear any existing cache
                torch.cuda.empty_cache()
                gc.collect()
                
                # Enable optimizations for RTX 4000 Ada
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                
                return 'cuda'
            else:
                return 'cpu'
        elif device_preference == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            return 'cuda'
        else:
            return 'cpu'
    
    def _print_init_info(self):
        """Print initialization information"""
        print("\n" + "="*70)
        print("ENHANCED SENTIMENT CLASSIFIER V2.0")
        print("="*70)
        print(f"Device: {self.device.upper()}")
        
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"VRAM: {total_memory:.1f} GB")
            print("✅ GPU acceleration enabled")
        else:
            print("⚠️ Running on CPU (slower performance)")
        
        print("="*70)
    
    def _load_models_improved(self):
        """Load models with improved error handling and fallback"""
        
        # Model configurations with fallback options
        model_configs = [
            {
                'name': 'primary',
                'model_id': 'lxyuan/distilbert-base-multilingual-cased-sentiments-student',
                'fallback_id': 'distilbert-base-uncased-finetuned-sst-2-english',
                'weight': 1.0,
                'max_length': 512
            }
        ]
        
        if self.use_ensemble:
            model_configs.append({
                'name': 'secondary',
                'model_id': 'cardiffnlp/twitter-roberta-base-sentiment',
                'fallback_id': 'distilbert-base-uncased-finetuned-sst-2-english',
                'weight': 0.7,
                'max_length': 512
            })
        
        print(f"\nLoading {len(model_configs)} model(s)...")
        successful_loads = 0
        
        for config in model_configs:
            loaded = self._load_single_model(config)
            if loaded:
                successful_loads += 1
        
        # If no models loaded, use rule-based fallback
        if successful_loads == 0:
            print("\n⚠️ All models failed to load. Using rule-based fallback.")
            self._setup_rule_based_fallback()
        else:
            print(f"\n✅ Successfully loaded {successful_loads} model(s)")
            self._normalize_model_weights()
    
    def _load_single_model(self, config: Dict) -> bool:
        """Load a single model with fallback options"""
        print(f"\nLoading {config['name']} model...")
        
        # Try primary model first
        model_loaded = self._try_load_model(
            config['model_id'], 
            config['name'],
            config['weight'],
            config['max_length']
        )
        
        if not model_loaded and config.get('fallback_id'):
            print(f"  Trying fallback model...")
            model_loaded = self._try_load_model(
                config['fallback_id'],
                config['name'],
                config['weight'],
                config['max_length']
            )
        
        return model_loaded
    
    def _try_load_model(self, model_id: str, name: str, weight: float, max_length: int) -> bool:
        """Try to load a specific model"""
        loading_strategies = []
        
        if self.device == 'cuda':
            # Try GPU with different precision levels
            loading_strategies = [
                ('GPU FP16', lambda: self._load_gpu_fp16(model_id, max_length)),
                ('GPU FP32', lambda: self._load_gpu_fp32(model_id, max_length)),
                ('CPU', lambda: self._load_cpu(model_id, max_length))
            ]
        else:
            # CPU only
            loading_strategies = [
                ('CPU', lambda: self._load_cpu(model_id, max_length))
            ]
        
        for strategy_name, strategy_func in loading_strategies:
            try:
                print(f"  Attempting {strategy_name} loading...")
                start_time = time.time()
                
                model_pipeline = strategy_func()
                
                if model_pipeline is not None:
                    load_time = time.time() - start_time
                    print(f"  ✅ Loaded via {strategy_name} in {load_time:.1f}s")
                    
                    self.models[name] = {
                        'pipeline': model_pipeline,
                        'weight': weight,
                        'model_id': model_id,
                        'device': strategy_name
                    }
                    
                    if 'GPU' in strategy_name:
                        self._print_gpu_memory()
                    
                    return True
                    
            except Exception as e:
                print(f"  ❌ {strategy_name} failed: {str(e)[:100]}")
                continue
        
        return False
    
    def _load_gpu_fp16(self, model_id: str, max_length: int):
        """Load model on GPU with FP16 precision"""
        return pipeline(
            'sentiment-analysis',
            model=model_id,
            device=0,
            torch_dtype=torch.float16,
            max_length=max_length,
            truncation=True
        )
    
    def _load_gpu_fp32(self, model_id: str, max_length: int):
        """Load model on GPU with FP32 precision"""
        return pipeline(
            'sentiment-analysis',
            model=model_id,
            device=0,
            torch_dtype=torch.float32,
            max_length=max_length,
            truncation=True
        )
    
    def _load_cpu(self, model_id: str, max_length: int):
        """Load model on CPU"""
        return pipeline(
            'sentiment-analysis',
            model=model_id,
            device=-1,
            max_length=max_length,
            truncation=True
        )
    
    def _print_gpu_memory(self):
        """Print current GPU memory usage"""
        if self.device == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def _setup_rule_based_fallback(self):
        """Setup rule-based sentiment analysis as fallback"""
        self.models['rule_based'] = {
            'pipeline': None,  # No actual pipeline for rule-based
            'weight': 1.0,
            'model_id': 'rule_based',
            'device': 'CPU'
        }
    
    def _normalize_model_weights(self):
        """Normalize model weights to sum to 1.0"""
        if not self.models:
            return
        
        total_weight = sum(m['weight'] for m in self.models.values())
        if total_weight > 0:
            for name in self.models:
                self.models[name]['weight'] /= total_weight
    
    def analyze_sentiment(self, text: str, language: str = 'auto') -> Dict:
        """
        Analyze sentiment with proper confidence normalization
        Returns sentiment and confidence in 0-1 range
        """
        if not text or not text.strip():
            return self._neutral_result()
        
        # Check cache first
        cache_key = hash(text[:100])  # Use first 100 chars for cache key
        if cache_key in self.cache:
            cached_result = self.cache[cache_key].copy()
            cached_result['from_cache'] = True
            return cached_result
        
        start_time = time.time()
        
        # Get predictions from all models
        predictions = self._get_model_predictions(text)
        
        if not predictions:
            # Use rule-based fallback
            result = self._rule_based_analysis(text)
        else:
            # Combine predictions
            result = self._combine_predictions(predictions)
        
        # Add metadata
        processing_time = time.time() - start_time
        result.update({
            'language': self._detect_language(text) if language == 'auto' else language,
            'processing_time': processing_time,
            'device': self.device,
            'models_used': len(predictions),
            'from_cache': False
        })
        
        # Ensure confidence is normalized (0-1 range)
        result['confidence'] = min(1.0, max(0.0, result['confidence']))
        
        # Cache result
        self._add_to_cache(cache_key, result)
        
        return result
    
    def _get_model_predictions(self, text: str) -> Dict:
        """Get predictions from all loaded models"""
        predictions = {}
        
        for name, model_info in self.models.items():
            if model_info['pipeline'] is None:
                # Rule-based model
                continue
            
            try:
                with torch.no_grad():
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    
                    result = model_info['pipeline'](text)
                    predictions[name] = {
                        'result': result,
                        'weight': model_info['weight']
                    }
                    
            except Exception as e:
                self.logger.warning(f"Model {name} prediction failed: {e}")
                continue
        
        return predictions
    
    def _combine_predictions(self, predictions: Dict) -> Dict:
        """Combine multiple model predictions with proper normalization"""
        if not predictions:
            return self._neutral_result()
        
        # Initialize score accumulators
        sentiment_scores = {
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 0.0
        }
        
        total_weight = 0.0
        
        for model_name, pred_info in predictions.items():
            result = pred_info['result']
            weight = pred_info['weight']
            
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            # Parse the prediction
            label = result.get('label', '').lower()
            score = min(1.0, max(0.0, result.get('score', 0.5)))  # Ensure 0-1 range
            
            # Map label to sentiment
            if any(pos in label for pos in ['positive', 'pos', '5 stars', '4 stars']):
                sentiment_scores['positive'] += score * weight
                sentiment_scores['negative'] += (1 - score) * weight * 0.5
                sentiment_scores['neutral'] += (1 - score) * weight * 0.5
            elif any(neg in label for neg in ['negative', 'neg', '1 star', '2 stars']):
                sentiment_scores['negative'] += score * weight
                sentiment_scores['positive'] += (1 - score) * weight * 0.5
                sentiment_scores['neutral'] += (1 - score) * weight * 0.5
            else:  # neutral
                sentiment_scores['neutral'] += score * weight
                sentiment_scores['positive'] += (1 - score) * weight * 0.5
                sentiment_scores['negative'] += (1 - score) * weight * 0.5
            
            total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            for sentiment in sentiment_scores:
                sentiment_scores[sentiment] /= total_weight
        
        # Determine final sentiment
        max_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        confidence = sentiment_scores[max_sentiment]
        
        # Ensure confidence is in valid range
        confidence = min(1.0, max(0.0, confidence))
        
        return {
            'sentiment': max_sentiment,
            'confidence': confidence,
            'scores': sentiment_scores
        }
    
    def _rule_based_analysis(self, text: str) -> Dict:
        """Enhanced rule-based sentiment analysis"""
        text_lower = text.lower()
        
        # Extended word lists
        strong_positive = [
            'excellent', 'amazing', 'fantastic', 'perfect', 'love', 'best',
            'awesome', 'outstanding', 'brilliant', 'superb', 'wonderful'
        ]
        
        moderate_positive = [
            'good', 'great', 'nice', 'like', 'helpful', 'useful', 'works',
            'satisfied', 'happy', 'recommend', 'smooth', 'easy'
        ]
        
        strong_negative = [
            'terrible', 'horrible', 'awful', 'worst', 'hate', 'useless',
            'garbage', 'trash', 'disaster', 'nightmare', 'broken'
        ]
        
        moderate_negative = [
            'bad', 'poor', 'disappointing', 'frustrated', 'annoying', 'slow',
            'difficult', 'confusing', 'problem', 'issue', 'bug', 'crash'
        ]
        
        # Count occurrences
        strong_pos_count = sum(1 for word in strong_positive if word in text_lower)
        moderate_pos_count = sum(1 for word in moderate_positive if word in text_lower)
        strong_neg_count = sum(1 for word in strong_negative if word in text_lower)
        moderate_neg_count = sum(1 for word in moderate_negative if word in text_lower)
        
        # Calculate weighted scores
        pos_score = (strong_pos_count * 2 + moderate_pos_count) / max(len(text.split()), 1)
        neg_score = (strong_neg_count * 2 + moderate_neg_count) / max(len(text.split()), 1)
        
        # Determine sentiment
        if pos_score > neg_score * 1.5:
            confidence = min(0.9, 0.5 + pos_score * 2)
            return {
                'sentiment': 'positive',
                'confidence': confidence,
                'scores': {
                    'positive': confidence,
                    'negative': (1 - confidence) * 0.3,
                    'neutral': (1 - confidence) * 0.7
                }
            }
        elif neg_score > pos_score * 1.5:
            confidence = min(0.9, 0.5 + neg_score * 2)
            return {
                'sentiment': 'negative',
                'confidence': confidence,
                'scores': {
                    'negative': confidence,
                    'positive': (1 - confidence) * 0.3,
                    'neutral': (1 - confidence) * 0.7
                }
            }
        else:
            confidence = 0.4 + min(0.4, (pos_score + neg_score))
            return {
                'sentiment': 'neutral',
                'confidence': confidence,
                'scores': {
                    'neutral': confidence,
                    'positive': (1 - confidence) / 2,
                    'negative': (1 - confidence) / 2
                }
            }
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns"""
        # This is a simplified version - you could use langdetect library
        if any(ord(char) > 127 for char in text):
            return 'multilingual'
        return 'en'
    
    def _add_to_cache(self, key: int, result: Dict):
        """Add result to cache with size limit"""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = result.copy()
    
    def _neutral_result(self) -> Dict:
        """Return neutral result for empty/invalid input"""
        return {
            'sentiment': 'neutral',
            'confidence': 0.0,
            'scores': {
                'positive': 0.33,
                'neutral': 0.34,
                'negative': 0.33
            }
        }
    
    def analyze_batch(self, texts: List[str], batch_size: int = None) -> List[Dict]:
        """
        Analyze multiple texts efficiently
        Returns list of sentiment results
        """
        if not texts:
            return []
        
        # Determine batch size based on device
        if batch_size is None:
            batch_size = 32 if self.device == 'cuda' else 8
        
        results = []
        total_texts = len(texts)
        
        print(f"\n📊 Batch processing {total_texts} texts...")
        start_time = time.time()
        
        for i in range(0, total_texts, batch_size):
            batch = texts[i:i+batch_size]
            batch_results = []
            
            # Process batch
            for text in batch:
                result = self.analyze_sentiment(text)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Progress update
            processed = min(i + batch_size, total_texts)
            if total_texts > 100 and processed % 100 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                print(f"  Progress: {processed}/{total_texts} ({rate:.1f} texts/sec)")
        
        total_time = time.time() - start_time
        rate = total_texts / total_time
        
        print(f"✅ Processed {total_texts} texts in {total_time:.1f}s ({rate:.1f} texts/sec)")
        
        # Clear GPU cache if used
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        info = {
            'version': '2.0',
            'device': self.device,
            'models_loaded': len(self.models),
            'models': {}
        }
        
        for name, model_info in self.models.items():
            info['models'][name] = {
                'model_id': model_info['model_id'],
                'weight': model_info['weight'],
                'device': model_info.get('device', 'unknown')
            }
        
        if self.device == 'cuda':
            info['gpu_memory'] = {
                'allocated': f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                'reserved': f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            }
        
        return info
    
    def cleanup(self):
        """Clean up resources"""
        # Clear cache
        self.cache.clear()
        
        # Clear GPU memory if using CUDA
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Clear models
        self.models.clear()
        
        # Force garbage collection
        gc.collect()
        
        print("✅ Cleanup completed")


# Integration helper class
class SentimentAspectIntegrator:
    """Helper class to integrate sentiment and aspect classifiers"""
    
    def __init__(self, sentiment_classifier, aspect_classifier):
        self.sentiment_classifier = sentiment_classifier
        self.aspect_classifier = aspect_classifier
    
    def analyze_complete(self, text: str) -> Dict:
        """
        Complete analysis with both sentiment and aspects
        Ensures proper data flow between classifiers
        """
        # Step 1: Sentiment analysis
        sentiment_result = self.sentiment_classifier.analyze_sentiment(text)
        
        # Step 2: Aspect analysis with sentiment context
        aspect_result = self.aspect_classifier.classify_aspects_multilabel(
            text=text,
            language=sentiment_result.get('language', 'en'),
            sentiment=sentiment_result['sentiment'],
            sentiment_confidence=sentiment_result['confidence']
        )
        
        # Step 3: Combine results
        return {
            'text': text,
            'sentiment': {
                'label': sentiment_result['sentiment'],
                'confidence': sentiment_result['confidence'],
                'confidence_percentage': f"{sentiment_result['confidence'] * 100:.1f}%",
                'scores': sentiment_result['scores']
            },
            'aspects': {
                'primary': aspect_result['primary_aspect'],
                'secondary': aspect_result['secondary_aspects'],
                'classification_type': aspect_result['classification_type'],
                'confidence': aspect_result['confidence'],
                'confidence_percentage': aspect_result['confidence_percentage']
            },
            'business_intelligence': {
                'priority_level': aspect_result['priority_level'],
                'severity_level': aspect_result['severity_level'],
                'summary': aspect_result['business_summary'],
                'recommendation': aspect_result['recommendation'],
                'requires_action': aspect_result['requires_immediate_action']
            },
            'metadata': {
                'processing_time': sentiment_result.get('processing_time', 0),
                'device': sentiment_result.get('device', 'unknown'),
                'models_used': sentiment_result.get('models_used', 0)
            }
        }


# Test if running directly
if __name__ == "__main__":
    print("🚀 Testing Enhanced Sentiment Classifier V2.0")
    print("="*70)
    
    # Initialize classifier
    classifier = EnhancedSentimentClassifier(use_ensemble=False)
    
    # Test cases
    test_texts = [
        "The app works so good I want to recommend it to all my colleagues.",
        "This app is absolutely terrible and keeps crashing!",
        "It's okay, nothing special but does the job.",
        "not receiving email for sign in, this app continues to be trash!",
        "Love the new features, much better than before!"
    ]
    
    print("\n📊 Sentiment Analysis Results:")
    print("-" * 70)
    
    for i, text in enumerate(test_texts, 1):
        result = classifier.analyze_sentiment(text)
        
        print(f"\n{i}. Text: '{text[:60]}...'" if len(text) > 60 else f"\n{i}. Text: '{text}'")
        print(f"   Sentiment: {result['sentiment'].upper()}")
        print(f"   Confidence: {result['confidence']*100:.1f}% ✅")  # Now properly normalized
        print(f"   Scores: Pos={result['scores']['positive']:.2f}, "
              f"Neg={result['scores']['negative']:.2f}, "
              f"Neu={result['scores']['neutral']:.2f}")
        print(f"   Device: {result['device']}, Time: {result['processing_time']*1000:.1f}ms")
    
    # Print model info
    print("\n" + "="*70)
    print("Model Information:")
    info = classifier.get_model_info()
    print(f"Version: {info['version']}")
    print(f"Device: {info['device']}")
    print(f"Models loaded: {info['models_loaded']}")
    for name, model_info in info['models'].items():
        print(f"  - {name}: {model_info['model_id']} (weight: {model_info['weight']:.2f})")
    
    print("\n✅ Enhanced Sentiment Classifier V2.0 ready for production!")
    print("✅ All confidence scores properly normalized (0-100%)!")
    
    # Cleanup
    classifier.cleanup()