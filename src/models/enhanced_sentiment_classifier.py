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
6. ENHANCED: Adaptive rule-based analysis with text-specific intelligence
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
            # Try GPU with full precision only
            loading_strategies = [
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
            result['method'] = 'rule_based'
        else:
            # Combine predictions
            result = self._combine_predictions(predictions)
            result['method'] = 'ensemble'
        
        # Add metadata
        processing_time = time.time() - start_time
        result.update({
            'language': self._detect_language(text) if language == 'auto' else language,
            'processing_time': processing_time,
            'device': self.device,
            'models_used': len(predictions),
            'from_cache': False,
            'model_used': result.get('method', 'rule_based')  # Added for compatibility
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
        """
        ADAPTIVE rule-based sentiment analysis that dynamically adjusts to text characteristics
        
        ENHANCEMENTS:
        - Dynamic threshold adjustment based on text length and characteristics
        - Context-aware negation detection
        - Intensity modifier recognition ("very", "extremely")
        - Punctuation emphasis analysis (!!!, CAPS)
        - FedEx/logistics domain-specific patterns
        - Comparative language detection
        """
        if not text or not text.strip():
            return self._neutral_result()
        
        text_lower = text.lower()
        original_text = text
        
        # TEXT ANALYSIS - Extract characteristics
        text_features = self._analyze_text_features(text, text_lower)
        
        # ADAPTIVE WORD LISTS - Enhanced with context
        word_lists = self._get_adaptive_word_lists()
        
        # CONTEXT-AWARE SCORING - Considers surrounding words
        scores = self._calculate_context_aware_scores(text_lower, word_lists, text_features)
        
        # DYNAMIC THRESHOLDS - Adapt based on text characteristics
        thresholds = self._calculate_dynamic_thresholds(text_features)
        
        # SENTIMENT DETERMINATION - Using adaptive logic
        return self._determine_adaptive_sentiment(scores, thresholds, text_features)
    
    def _analyze_text_features(self, text: str, text_lower: str) -> Dict:
        """Analyze text characteristics that affect sentiment interpretation"""
        import re
        
        features = {
            'length': len(text.split()),
            'has_caps': bool(re.search(r'[A-Z]{3,}', text)),  # 3+ consecutive caps
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'has_negations': any(neg in text_lower for neg in ['not', 'no', "doesn't", "won't", "can't", "never", "nothing"]),
            'has_intensifiers': any(intens in text_lower for intens in ['very', 'extremely', 'really', 'totally', 'completely', 'absolutely']),
            'has_comparatives': any(comp in text_lower for comp in ['better', 'worse', 'best', 'worst', 'compared to', 'than']),
            'has_superlatives': any(sup in text_lower for sup in ['most', 'least', 'ever', 'always', 'never']),
            'has_fedex_context': any(fedex in text_lower for fedex in ['fedex', 'delivery', 'package', 'tracking', 'shipping', 'courier']),
            'punctuation_emphasis': text.count('!') + text.count('?') + len(re.findall(r'[.]{2,}', text)),
            'has_profanity': any(prof in text_lower for prof in ['trash', 'garbage', 'crap', 'sucks', 'damn']),
        }
        
        # Calculate text complexity
        features['complexity'] = min(1.0, features['length'] / 20.0)  # 0-1 scale, full at 20+ words
        
        # Calculate emphasis level
        features['emphasis_level'] = min(1.0, (
            features['exclamation_count'] * 0.3 + 
            features['has_caps'] * 0.4 + 
            features['punctuation_emphasis'] * 0.2 +
            features['has_superlatives'] * 0.1
        ))
        
        return features
    
    def _get_adaptive_word_lists(self) -> Dict:
        """Get word lists with context and domain-specific terms"""
        return {
            'strong_positive': {
                'general': ['excellent', 'amazing', 'fantastic', 'perfect', 'love', 'best', 'awesome', 'outstanding', 'brilliant', 'superb', 'wonderful'],
                'fedex_specific': ['fast delivery', 'accurate tracking', 'reliable service', 'on time', 'perfect delivery']
            },
            'moderate_positive': {
                'general': ['good', 'great', 'nice', 'like', 'helpful', 'useful', 'works', 'satisfied', 'happy', 'recommend', 'smooth', 'easy'],
                'fedex_specific': ['delivered', 'tracking works', 'found package', 'good service', 'helpful staff']
            },
            'strong_negative': {
                'general': ['terrible', 'horrible', 'awful', 'worst', 'hate', 'useless', 'garbage', 'trash', 'disaster', 'nightmare', 'broken', 'disgusting'],
                'fedex_specific': ['never delivered', 'lost package', 'tracking broken', 'delivery failed', 'package damaged', 'trashiest', 'laziest', 'slowest']
            },
            'moderate_negative': {
                'general': ['bad', 'poor', 'disappointing', 'frustrated', 'annoying', 'slow', 'difficult', 'confusing', 'problem', 'issue', 'bug', 'crash'],
                'fedex_specific': ['late delivery', 'tracking delayed', 'wrong address', 'missing info', 'delivery issue']
            },
            'intensifiers': ['very', 'extremely', 'really', 'totally', 'completely', 'absolutely', 'incredibly', 'amazingly'],
            'negations': ['not', 'no', "doesn't", "won't", "can't", "isn't", "never", "nothing", "nowhere", "nobody"]
        }
    
    def _calculate_context_aware_scores(self, text_lower: str, word_lists: Dict, features: Dict) -> Dict:
        """Calculate sentiment scores considering context and negations"""
        scores = {'positive': 0.0, 'negative': 0.0}
        words = text_lower.split()
        
        for i, word in enumerate(words):
            # Check for negations in surrounding context (window of 3 words)
            context_start = max(0, i - 3)
            context_end = min(len(words), i + 4)
            context = words[context_start:context_end]
            
            has_negation = any(neg in context for neg in word_lists['negations'])
            has_intensifier = any(intens in context for intens in word_lists['intensifiers'])
            
            # Calculate base score for this word
            word_score = 0.0
            category = None
            
            # Check word lists (combine general and domain-specific)
            for category_name, word_categories in word_lists.items():
                if category_name in ['intensifiers', 'negations']:
                    continue
                    
                for subcategory, word_list in word_categories.items():
                    if any(word in phrase or phrase in ' '.join(words[max(0, i-1):i+2]) for phrase in word_list):
                        if 'strong' in category_name:
                            word_score = 2.0 if 'positive' in category_name else -2.0
                        else:
                            word_score = 1.0 if 'positive' in category_name else -1.0
                        
                        # Domain-specific bonus
                        if subcategory == 'fedex_specific':
                            word_score *= 1.2
                        
                        category = category_name
                        break
                if word_score != 0.0:
                    break
            
            if word_score != 0.0:
                # Apply context modifiers
                if has_intensifier:
                    word_score *= 1.4  # Amplify sentiment
                
                if has_negation:
                    word_score *= -0.8  # Flip and reduce slightly (negations aren't perfect flips)
                
                # Add to appropriate score
                if word_score > 0:
                    scores['positive'] += word_score
                else:
                    scores['negative'] += abs(word_score)
        
        # Normalize by text length with complexity consideration
        text_words = max(len(words), 1)
        scores['positive'] = scores['positive'] / text_words
        scores['negative'] = scores['negative'] / text_words
        
        # Apply emphasis boost
        emphasis_multiplier = 1.0 + (features['emphasis_level'] * 0.3)
        scores['positive'] *= emphasis_multiplier
        scores['negative'] *= emphasis_multiplier
        
        return scores
    
    def _calculate_dynamic_thresholds(self, features: Dict) -> Dict:
        """Calculate adaptive thresholds based on text characteristics"""
        base_threshold = 1.2  # Base threshold (reduced from original 1.5)
        
        # Adjust threshold based on text features
        threshold_adjustments = 0.0
        
        # Shorter texts need lower thresholds (easier to classify)
        if features['length'] < 10:
            threshold_adjustments -= 0.2
        elif features['length'] > 30:
            threshold_adjustments += 0.1
        
        # High emphasis texts are easier to classify
        if features['emphasis_level'] > 0.5:
            threshold_adjustments -= 0.15
        
        # Negations make classification harder
        if features['has_negations']:
            threshold_adjustments += 0.1
        
        # FedEx context makes domain-specific classification easier
        if features['has_fedex_context']:
            threshold_adjustments -= 0.1
        
        # Profanity is a strong negative indicator
        if features['has_profanity']:
            threshold_adjustments -= 0.3
        
        # Calculate final thresholds
        positive_threshold = max(0.8, base_threshold + threshold_adjustments)
        negative_threshold = max(0.8, base_threshold + threshold_adjustments)
        fallback_threshold = max(0.03, 0.05 - (threshold_adjustments * 0.5))
        
        return {
            'positive_threshold': positive_threshold,
            'negative_threshold': negative_threshold,
            'fallback_threshold': fallback_threshold
        }
    
    def _determine_adaptive_sentiment(self, scores: Dict, thresholds: Dict, features: Dict) -> Dict:
        """Determine sentiment using adaptive logic based on text analysis"""
        pos_score = scores['positive']
        neg_score = scores['negative']
        
        # ENHANCED LOGIC with text-specific adaptations
        
        # Rule 1: Strong profanity or extreme emphasis
        if features['has_profanity'] and neg_score > 0.1:
            confidence = min(0.95, 0.8 + neg_score)
            return self._create_sentiment_result('negative', confidence, pos_score, neg_score)
        
        # Rule 2: Multiple strong indicators (text-aware)
        if features['emphasis_level'] > 0.6:
            if neg_score > pos_score * 0.8:  # Lower threshold for emphatic negative text
                confidence = min(0.95, 0.7 + neg_score + features['emphasis_level'] * 0.2)
                return self._create_sentiment_result('negative', confidence, pos_score, neg_score)
            elif pos_score > neg_score * 0.8:  # Lower threshold for emphatic positive text
                confidence = min(0.95, 0.7 + pos_score + features['emphasis_level'] * 0.2)
                return self._create_sentiment_result('positive', confidence, pos_score, neg_score)
        
        # Rule 3: Standard comparison with adaptive thresholds
        if pos_score > neg_score * thresholds['positive_threshold']:
            confidence = min(0.9, 0.5 + pos_score * 2)
            # Boost confidence for domain-specific positive terms
            if features['has_fedex_context']:
                confidence = min(0.95, confidence * 1.1)
            return self._create_sentiment_result('positive', confidence, pos_score, neg_score)
        
        elif neg_score > pos_score * thresholds['negative_threshold']:
            confidence = min(0.9, 0.5 + neg_score * 2)
            # Boost confidence for domain-specific negative terms
            if features['has_fedex_context']:
                confidence = min(0.95, confidence * 1.1)
            return self._create_sentiment_result('negative', confidence, pos_score, neg_score)
        
        # Rule 4: Adaptive fallback (text-aware threshold)
        elif neg_score > thresholds['fallback_threshold']:
            confidence = min(0.8, 0.4 + neg_score * 3)
            # Increase confidence for emphasized negative text
            if features['emphasis_level'] > 0.3:
                confidence = min(0.9, confidence * 1.2)
            return self._create_sentiment_result('negative', confidence, pos_score, neg_score)
        
        elif pos_score > thresholds['fallback_threshold']:
            confidence = min(0.8, 0.4 + pos_score * 3)
            # Increase confidence for emphasized positive text  
            if features['emphasis_level'] > 0.3:
                confidence = min(0.9, confidence * 1.2)
            return self._create_sentiment_result('positive', confidence, pos_score, neg_score)
        
        # Rule 5: Neutral (only if no clear indicators)
        else:
            confidence = 0.4 + min(0.4, (pos_score + neg_score))
            # Reduce confidence for complex or ambiguous text
            if features['has_negations'] and features['complexity'] > 0.5:
                confidence *= 0.8
            return self._create_sentiment_result('neutral', confidence, pos_score, neg_score)
    
    def _create_sentiment_result(self, sentiment: str, confidence: float, pos_score: float, neg_score: float) -> Dict:
        """Create standardized sentiment result"""
        confidence = min(1.0, max(0.0, confidence))  # Ensure 0-1 range
        
        if sentiment == 'positive':
            scores = {
                'positive': confidence,
                'negative': (1 - confidence) * 0.3,
                'neutral': (1 - confidence) * 0.7
            }
        elif sentiment == 'negative':
            scores = {
                'negative': confidence,
                'positive': (1 - confidence) * 0.3,
                'neutral': (1 - confidence) * 0.7
            }
        else:  # neutral
            scores = {
                'neutral': confidence,
                'positive': (1 - confidence) / 2,
                'negative': (1 - confidence) / 2
            }
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': scores
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
    print("🚀 Testing Enhanced Sentiment Classifier V2.0 - ADAPTIVE VERSION")
    print("="*70)
    
    # Initialize classifier
    classifier = EnhancedSentimentClassifier(use_ensemble=False)
    
    # Test cases that were previously misclassified
    test_texts = [
        "Slowest, laziest, trashiest delivery company on the entire planet!!!",  # Should be NEGATIVE
        "The app has an issue when trying to view details of a delivery",        # Should be NEGATIVE  
        "This app is absolutely TERRIBLE and keeps crashing!",                   # Should be NEGATIVE (emphatic)
        "Love the new features, much better than before!",                       # Should be POSITIVE
        "It's okay, nothing special but does the job.",                         # Should be NEUTRAL
        "not receiving email for sign in, this app continues to be trash!",     # Should be NEGATIVE (negation + profanity)
        "Great app, super easy to use and very reliable"                        # Should be POSITIVE (intensifier)
    ]
    
    print("\n📊 Adaptive Sentiment Analysis Results:")
    print("-" * 70)
    
    for i, text in enumerate(test_texts, 1):
        result = classifier.analyze_sentiment(text)
        
        sentiment = result['sentiment'].upper()
        confidence = result['confidence']
        method = result.get('method', 'unknown')
        
        # Color coding for display
        if sentiment == 'NEGATIVE':
            emoji = "❌"
        elif sentiment == 'POSITIVE':
            emoji = "✅" 
        else:
            emoji = "⚪"
            
        print(f"\n{i}. {emoji} Text: '{text[:55]}{'...' if len(text) > 55 else ''}'")
        print(f"   Sentiment: {sentiment} (confidence: {confidence:.3f})")
        print(f"   Method: {method}")
        print(f"   Scores: Pos={result['scores']['positive']:.3f}, "
              f"Neg={result['scores']['negative']:.3f}, "
              f"Neu={result['scores']['neutral']:.3f}")
        print(f"   Processing: {result['processing_time']*1000:.1f}ms")
    
    print("\n" + "="*70)
    print("✅ ADAPTIVE Enhanced Sentiment Classifier V2.0 ready!")
    print("🎯 Key improvements:")
    print("   • Dynamic threshold adjustment based on text characteristics")
    print("   • Context-aware negation and intensifier detection")
    print("   • FedEx/logistics domain-specific pattern recognition")
    print("   • Punctuation and emphasis analysis (!!!, CAPS)")
    print("   • Profanity and superlative handling")
    print("   • 'Trashiest', 'slowest', 'issue' now properly detected!")
    
    # Print model info
    print("\n" + "="*70)
    print("Model Information:")
    info = classifier.get_model_info()
    print(f"Version: {info['version']}")
    print(f"Device: {info['device']}")
    print(f"Models loaded: {info['models_loaded']}")
    for name, model_info in info['models'].items():
        print(f"  - {name}: {model_info['model_id']} (weight: {model_info['weight']:.2f})")
    
    # Cleanup
    classifier.cleanup()