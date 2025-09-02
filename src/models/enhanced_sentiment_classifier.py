#!/usr/bin/env python3
"""
Enhanced Sentiment Classifier - Two-Model Ensemble with Advanced Features
XLM-RoBERTa + Twitter-RoBERTa ensemble with restored advanced functionality
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
import re

warnings.filterwarnings('ignore')

class EnhancedSentimentClassifier:
    """
    Enhanced Sentiment Classifier with Two-Model Ensemble
    XLM-RoBERTa (53.3%) + Twitter-RoBERTa (46.7%) + Advanced Features
    """
    
    def __init__(self, device='auto', verbose=True):
        self.verbose = verbose
        self.models = {}
        self.device = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Configure device
        self.device = self._configure_device(device)
        
        # Print initialization info
        if self.verbose:
            self._print_init_info()
        
        # Load the two-model ensemble
        self._load_two_model_ensemble()
        
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
                
                # Enable optimizations for GPU
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
        print("ENHANCED SENTIMENT CLASSIFIER - TWO-MODEL ENSEMBLE")
        print("="*70)
        print(f"Device: {self.device.upper()}")
        
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"VRAM: {total_memory:.1f} GB")
            print("GPU acceleration enabled")
        else:
            print("Running on CPU")
        
        print("Models: XLM-RoBERTa (53.3%) + Twitter-RoBERTa (46.7%)")
        print("="*70)
    
    def _load_two_model_ensemble(self):
        """Load the optimized two-model ensemble"""
        
        # Two-model configuration with tested weights
        model_configs = [
            {
                'name': 'xlm_roberta',
                'model_id': 'cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual',
                'fallback_id': 'distilbert-base-uncased-finetuned-sst-2-english',
                'weight': 0.533,  # 53.3% - primary model
                'max_length': 512
            },
            {
                'name': 'twitter_roberta',
                'model_id': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'fallback_id': 'distilbert-base-uncased-finetuned-sst-2-english', 
                'weight': 0.467,  # 46.7% - secondary model
                'max_length': 512
            }
        ]
        
        if self.verbose:
            print(f"\nLoading two-model ensemble...")
        
        successful_loads = 0
        
        for config in model_configs:
            loaded = self._load_single_model(config)
            if loaded:
                successful_loads += 1
        
        # Ensure we have at least one model
        if successful_loads == 0:
            if self.verbose:
                print("\nAll models failed to load. Using rule-based fallback.")
            self._setup_rule_based_fallback()
        else:
            if self.verbose:
                print(f"\nSuccessfully loaded {successful_loads}/2 ensemble models")
            self._normalize_model_weights()
    
    def _load_single_model(self, config: Dict) -> bool:
        """Load a single model with fallback options"""
        if self.verbose:
            print(f"\nLoading {config['name']} model...")
        
        # Try primary model first
        model_loaded = self._try_load_model(
            config['model_id'], 
            config['name'],
            config['weight'],
            config['max_length']
        )
        
        if not model_loaded and config.get('fallback_id'):
            if self.verbose:
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
            # Only FP32 strategies (no FP16 as requested)
            loading_strategies = [
                ('GPU FP32', lambda: self._load_gpu_fp32(model_id, max_length)),
                ('CPU', lambda: self._load_cpu(model_id, max_length))
            ]
        else:
            loading_strategies = [
                ('CPU', lambda: self._load_cpu(model_id, max_length))
            ]
        
        for strategy_name, strategy_func in loading_strategies:
            try:
                if self.verbose:
                    print(f"  Attempting {strategy_name} loading...")
                start_time = time.time()
                
                model_pipeline = strategy_func()
                
                if model_pipeline is not None:
                    load_time = time.time() - start_time
                    if self.verbose:
                        print(f"  Loaded via {strategy_name} in {load_time:.1f}s")
                    
                    self.models[name] = {
                        'pipeline': model_pipeline,
                        'weight': weight,
                        'model_id': model_id,
                        'device': strategy_name
                    }
                    
                    if 'GPU' in strategy_name and self.verbose:
                        self._print_gpu_memory()
                    
                    return True
                    
            except Exception as e:
                if self.verbose:
                    print(f"  {strategy_name} failed: {str(e)[:100]}")
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
        
        if self.verbose:
            print("\nModel weights (normalized):")
            for name, model_info in self.models.items():
                print(f"  {name}: {model_info['weight']:.3f}")
    
    def analyze_sentiment(self, text: str, language: str = 'auto') -> Dict:
        """
        Analyze sentiment with two-model ensemble
        Returns sentiment and confidence in 0-1 range
        """
        if not text or not text.strip():
            return self._neutral_result()
        
        # Check cache first
        cache_key = hash(text[:100])
        if cache_key in self.cache:
            cached_result = self.cache[cache_key].copy()
            cached_result['from_cache'] = True
            return cached_result
        
        start_time = time.time()
        
        # Get predictions from ensemble models
        predictions = self._get_model_predictions(text)
        
        if not predictions:
            # Use enhanced rule-based fallback
            result = self._advanced_rule_based_analysis(text)
            result['method'] = 'rule_based'
            result['models_available'] = 0
        else:
            # Combine predictions from ensemble
            result = self._combine_predictions(predictions)
            result['method'] = 'two_model_ensemble'
            result['models_available'] = len(predictions)
        
        # Add metadata
        processing_time = time.time() - start_time
        result.update({
            'language': self._detect_language(text) if language == 'auto' else language,
            'processing_time': processing_time,
            'device': self.device,
            'models_used': len(predictions),
            'from_cache': False,
            'model_used': result.get('method', 'rule_based')
        })
        
        # Ensure confidence is normalized (0-1 range)
        result['confidence'] = min(1.0, max(0.0, result['confidence']))
        
        # Cache result
        self._add_to_cache(cache_key, result)
        
        return result
    
    def _get_model_predictions(self, text: str) -> Dict:
        """Get predictions from all loaded ensemble models"""
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
        """Combine multiple model predictions with weighted voting"""
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
            score = min(1.0, max(0.0, result.get('score', 0.5)))
            
            # Map label to sentiment with weighted voting
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
    
    def _advanced_rule_based_analysis(self, text: str) -> Dict:
        """
        ADVANCED rule-based sentiment analysis with multilingual support
        Enhanced fallback when transformer models fail
        """
        if not text or not text.strip():
            return self._neutral_result()
        
        text_lower = text.lower()
        original_text = text
        
        # Analyze text features for dynamic thresholding
        text_features = self._analyze_text_features(text, text_lower)
        
        # Get multilingual word lists
        word_lists = self._get_multilingual_word_lists()
        
        # Calculate context-aware scores
        scores = self._calculate_context_aware_scores(text_lower, word_lists, text_features)
        
        # Apply dynamic thresholds
        thresholds = self._calculate_dynamic_thresholds(text_features)
        
        # Determine final sentiment
        return self._determine_adaptive_sentiment(scores, thresholds, text_features)
    
    def _analyze_text_features(self, text: str, text_lower: str) -> Dict:
        """Analyze text characteristics for better sentiment detection"""
        features = {
            'length': len(text.split()),
            'has_caps': bool(re.search(r'[A-Z]{3,}', text)),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'has_negations': any(neg in text_lower for neg in ['not', 'no', "doesn't", "won't", "can't", "never", "nothing"]),
            'has_intensifiers': any(intens in text_lower for intens in ['very', 'extremely', 'really', 'totally', 'completely', 'absolutely']),
            'has_superlatives': any(sup in text_lower for sup in ['most', 'least', 'ever', 'always', 'never', 'best', 'worst']),
            'punctuation_emphasis': text.count('!') + text.count('?') + len(re.findall(r'[.]{2,}', text)),
            'has_profanity': any(prof in text_lower for prof in [
                'trash', 'garbage', 'crap', 'sucks', 'damn', 'shit', 'fuck', 
                'useless', 'terrible', 'horrible', 'awful', 'worst', 'hate', 'trashiest'
            ]),
            'original_text': text_lower
        }
        
        # Calculate emphasis level
        features['emphasis_level'] = min(1.0, (
            features['exclamation_count'] * 0.3 + 
            features['has_caps'] * 0.4 + 
            features['punctuation_emphasis'] * 0.2 +
            features['has_superlatives'] * 0.1
        ))
        
        return features
    
    def _get_multilingual_word_lists(self) -> Dict:
        """Enhanced multilingual word lists for better global coverage"""
        return {
            'strong_positive': {
                'english': ['excellent', 'amazing', 'fantastic', 'perfect', 'love', 'best', 'awesome', 'outstanding', 'brilliant', 'superb', 'wonderful'],
                'spanish': ['excelente', 'increíble', 'fantástico', 'perfecto', 'amo', 'mejor', 'genial', 'maravilloso'],
                'german': ['ausgezeichnet', 'erstaunlich', 'fantastisch', 'perfekt', 'liebe', 'beste', 'toll', 'wunderbar'],
                'french': ['excellent', 'incroyable', 'fantastique', 'parfait', 'amour', 'meilleur', 'génial', 'merveilleux'],
                'dutch': ['uitstekend', 'geweldig', 'fantastisch', 'perfect', 'houd van', 'beste', 'geweldig']
            },
            'strong_negative': {
                'english': ['terrible', 'horrible', 'awful', 'worst', 'hate', 'useless', 'garbage', 'trash', 'disaster', 'nightmare', 'broken', 'trashiest', 'laziest', 'slowest'],
                'spanish': ['terrible', 'horrible', 'horrible', 'peor', 'odio', 'inútil', 'basura', 'desastre'],
                'german': ['schrecklich', 'furchtbar', 'schlimm', 'schlechteste', 'hasse', 'nutzlos', 'müll', 'katastrophe'],
                'french': ['terrible', 'horrible', 'affreux', 'pire', 'déteste', 'inutile', 'ordures', 'désastre'],
                'dutch': ['verschrikkelijk', 'afschuwelijk', 'vreselijk', 'ergste', 'haat', 'nutteloos', 'afval']
            },
            'moderate_positive': {
                'english': ['good', 'great', 'nice', 'like', 'helpful', 'useful', 'works', 'satisfied', 'happy', 'recommend'],
                'spanish': ['bueno', 'genial', 'agradable', 'gusta', 'útil', 'funciona', 'satisfecho', 'feliz'],
                'german': ['gut', 'toll', 'schön', 'mag', 'hilfreich', 'nützlich', 'funktioniert', 'zufrieden'],
                'french': ['bon', 'super', 'agréable', 'aime', 'utile', 'marche', 'satisfait', 'heureux'],
                'dutch': ['goed', 'geweldig', 'leuk', 'vind leuk', 'handig', 'werkt', 'tevreden', 'blij']
            },
            'moderate_negative': {
                'english': ['bad', 'poor', 'disappointing', 'frustrated', 'annoying', 'slow', 'difficult', 'confusing', 'problem', 'issue', 'bug', 'crash', 'sucks'],
                'spanish': ['malo', 'pobre', 'decepcionante', 'frustrado', 'molesto', 'lento', 'difícil', 'problema'],
                'german': ['schlecht', 'arm', 'enttäuschend', 'frustriert', 'ärgerlich', 'langsam', 'schwierig', 'problem'],
                'french': ['mauvais', 'pauvre', 'décevant', 'frustré', 'ennuyeux', 'lent', 'difficile', 'problème'],
                'dutch': ['slecht', 'arm', 'teleurstellend', 'gefrustreerd', 'vervelend', 'langzaam', 'moeilijk', 'probleem']
            },
            'intensifiers': ['very', 'extremely', 'really', 'totally', 'completely', 'absolutely', 'incredibly', 'muy', 'sehr', 'très', 'heel'],
            'negations': ['not', 'no', "doesn't", "won't", "can't", "isn't", "never", 'nothing', 'nowhere', 'nobody', 'ningún', 'nicht', 'pas', 'niet']
        }
    
    def _calculate_context_aware_scores(self, text_lower: str, word_lists: Dict, features: Dict) -> Dict:
        """Calculate sentiment scores considering context and negations"""
        scores = {'positive': 0.0, 'negative': 0.0}
        words = text_lower.split()
        
        for i, word in enumerate(words):
            # Check for negations and intensifiers in context
            context_start = max(0, i - 3)
            context_end = min(len(words), i + 4)
            context = words[context_start:context_end]
            
            has_negation = any(neg in context for neg in word_lists['negations'])
            has_intensifier = any(intens in context for intens in word_lists['intensifiers'])
            
            # Calculate base score for this word across all languages
            word_score = 0.0
            
            for sentiment_type, lang_dict in word_lists.items():
                if sentiment_type in ['intensifiers', 'negations']:
                    continue
                    
                for lang, word_list in lang_dict.items():
                    if any(word in phrase or phrase in ' '.join(words[max(0, i-1):i+2]) for phrase in word_list):
                        if 'strong' in sentiment_type:
                            word_score = 2.0 if 'positive' in sentiment_type else -2.0
                        else:
                            word_score = 1.0 if 'positive' in sentiment_type else -1.0
                        break
                if word_score != 0.0:
                    break
            
            if word_score != 0.0:
                # Apply context modifiers
                if has_intensifier:
                    word_score *= 1.4
                
                if has_negation:
                    word_score *= -0.8
                
                # Add to appropriate score
                if word_score > 0:
                    scores['positive'] += word_score
                else:
                    scores['negative'] += abs(word_score)
        
        # Normalize by text length
        text_words = max(len(words), 1)
        scores['positive'] = scores['positive'] / text_words
        scores['negative'] = scores['negative'] / text_words
        
        # Apply emphasis boost
        emphasis_multiplier = 1.0 + (features['emphasis_level'] * 0.3)
        scores['positive'] *= emphasis_multiplier
        scores['negative'] *= emphasis_multiplier
        
        return scores
    
    def _calculate_dynamic_thresholds(self, features: Dict) -> Dict:
        """Calculate dynamic thresholds based on text characteristics"""
        base_threshold = 0.8
        
        threshold_adjustments = 0.0
        
        # Shorter texts need lower thresholds
        if features['length'] < 10:
            threshold_adjustments -= 0.3
        elif features['length'] > 30:
            threshold_adjustments += 0.1
        
        # High emphasis texts are easier to classify
        if features['emphasis_level'] > 0.5:
            threshold_adjustments -= 0.25
        
        # Profanity is a strong negative indicator
        if features['has_profanity']:
            threshold_adjustments -= 0.5
        
        positive_threshold = max(0.6, base_threshold + threshold_adjustments)
        negative_threshold = max(0.5, base_threshold + threshold_adjustments)
        
        return {
            'positive_threshold': positive_threshold,
            'negative_threshold': negative_threshold
        }
    
    def _determine_adaptive_sentiment(self, scores: Dict, thresholds: Dict, features: Dict) -> Dict:
        """Determine sentiment with adaptive logic"""
        pos_score = scores['positive']
        neg_score = scores['negative']
        
        # Check for critical negative indicators
        critical_negative_phrases = [
            'absolute trash', 'complete garbage', 'worst ever', 'total disaster',
            'piece of shit', 'piece of crap', 'never again', 'hate this app'
        ]
        text_lower = features.get('original_text', '')
        
        has_critical_negative = any(phrase in text_lower for phrase in critical_negative_phrases)
        if has_critical_negative or (features['has_profanity'] and neg_score > 0.05):
            confidence = min(0.95, 0.85 + neg_score)
            return self._create_sentiment_result('negative', confidence, pos_score, neg_score)
        
        # Apply threshold comparisons
        if neg_score > pos_score * thresholds['negative_threshold']:
            confidence = min(0.9, 0.6 + neg_score * 2.5)
            return self._create_sentiment_result('negative', confidence, pos_score, neg_score)
        elif pos_score > neg_score * thresholds['positive_threshold']:
            confidence = min(0.9, 0.5 + pos_score * 2)
            return self._create_sentiment_result('positive', confidence, pos_score, neg_score)
        else:
            confidence = 0.4 + min(0.3, max(pos_score, neg_score))
            return self._create_sentiment_result('neutral', confidence, pos_score, neg_score)
    
    def _create_sentiment_result(self, sentiment: str, confidence: float, pos_score: float, neg_score: float) -> Dict:
        """Create standardized sentiment result"""
        confidence = min(1.0, max(0.0, confidence))
        
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
        """Simple language detection"""
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
        Enhanced batch processing with progress tracking and memory management
        """
        if not texts:
            return []
        
        # Determine batch size based on device and memory
        if batch_size is None:
            if self.device == 'cuda':
                # Check available GPU memory
                if torch.cuda.is_available():
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                    # Adjust batch size based on available memory
                    if free_memory > 8 * 1024**3:  # > 8GB
                        batch_size = 64
                    elif free_memory > 4 * 1024**3:  # > 4GB
                        batch_size = 32
                    else:
                        batch_size = 16
                else:
                    batch_size = 32
            else:
                batch_size = 8
        
        results = []
        total_texts = len(texts)
        
        if self.verbose:
            print(f"\nBatch processing {total_texts} texts (batch_size={batch_size})...")
            
        start_time = time.time()
        processed_count = 0
        
        # Process in batches with progress tracking
        for batch_start in range(0, total_texts, batch_size):
            batch_end = min(batch_start + batch_size, total_texts)
            batch = texts[batch_start:batch_end]
            
            # Process current batch
            batch_results = []
            for text in batch:
                result = self.analyze_sentiment(text)
                batch_results.append(result)
            
            results.extend(batch_results)
            processed_count += len(batch_results)
            
            # Progress updates for large batches
            if self.verbose and (total_texts > 50):
                if processed_count % max(50, total_texts // 10) == 0 or processed_count == total_texts:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    eta = (total_texts - processed_count) / rate if rate > 0 else 0
                    
                    print(f"  Progress: {processed_count:,}/{total_texts:,} "
                          f"({processed_count/total_texts*100:.1f}%) - "
                          f"Rate: {rate:.1f} texts/sec - "
                          f"ETA: {eta:.1f}s")
            
            # Memory management for GPU
            if self.device == 'cuda' and batch_start % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
        
        # Final statistics
        total_time = time.time() - start_time
        avg_rate = total_texts / total_time if total_time > 0 else 0
        
        if self.verbose:
            print(f"Batch processing complete!")
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Average rate: {avg_rate:.1f} texts/sec")
            print(f"  Cache hits: {sum(1 for r in results if r.get('from_cache', False))}")
        
        # Final cleanup
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the two-model ensemble"""
        info = {
            'version': '2.0_two_model_ensemble_advanced',
            'device': self.device,
            'ensemble_enabled': True,
            'loaded_models': len([m for m in self.models.values() if m['pipeline'] is not None]),
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
        
        if self.verbose:
            print("Cleanup completed")


# Integration helper class for combined sentiment + aspect analysis
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
    
    def analyze_batch_integrated(self, texts: List[str]) -> List[Dict]:
        """Batch analysis with integrated sentiment and aspect classification"""
        results = []
        
        # Get sentiment results first
        sentiment_results = self.sentiment_classifier.analyze_batch(texts)
        
        # Process aspects with sentiment context
        for i, text in enumerate(texts):
            sentiment_data = sentiment_results[i] if i < len(sentiment_results) else {}
            
            # Get aspect classification
            aspect_result = self.aspect_classifier.classify_aspects_multilabel(
                text=text,
                sentiment=sentiment_data.get('sentiment', 'neutral'),
                sentiment_confidence=sentiment_data.get('confidence', 0.5)
            )
            
            # Combine results
            integrated_result = {
                'text': text,
                'sentiment': sentiment_data,
                'aspects': aspect_result,
                'integrated_analysis': True
            }
            
            results.append(integrated_result)
        
        return results


# Test the enhanced two-model ensemble
if __name__ == "__main__":
    print("Testing Enhanced Two-Model Ensemble with Advanced Features")
    print("="*70)
    
    # Initialize classifier
    classifier = EnhancedSentimentClassifier(verbose=True)
    
    # Test cases including challenging ones
    test_texts = [
        "not receiving email for sign in, this app continues to be trash!",
        "Slowest, laziest, trashiest delivery company on the entire planet!!!",
        "The app has an issue when trying to view details of a delivery",
        "Love the new features, much better than before!",
        "La aplicación es muy buena pero tiene problemas ocasionales",  # Spanish
        "Die App stürzt ständig ab, sehr frustrierend",  # German
        "Cette application est fantastique, je la recommande vivement!",  # French
        "Complete disaster, never works, worst app ever"
    ]
    
    print("\nTesting Enhanced Two-Model Ensemble:")
    print("-" * 70)
    
    for i, text in enumerate(test_texts, 1):
        result = classifier.analyze_sentiment(text)
        
        print(f"\n{i}. Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"   Sentiment: {result['sentiment'].upper()} ({result['confidence']:.3f})")
        print(f"   Method: {result['method']}")
        print(f"   Models Used: {result['models_used']}")
        print(f"   Language: {result['language']}")
        print(f"   Processing: {result['processing_time']*1000:.1f}ms")
    
    # Test batch processing
    print(f"\n" + "="*70)
    print("Testing Enhanced Batch Processing:")
    batch_results = classifier.analyze_batch(test_texts[:4], batch_size=2)
    print(f"Processed {len(batch_results)} texts in batch mode")
    
    # Print model info
    print("\n" + "="*70)
    print("Enhanced Two-Model Ensemble Information:")
    info = classifier.get_model_info()
    print(f"Version: {info['version']}")
    print(f"Device: {info['device']}")
    print(f"Models loaded: {info['loaded_models']}")
    for name, model_info in info['models'].items():
        if model_info['model_id'] != 'rule_based':
            print(f"  {name}: {model_info['model_id']} (weight: {model_info['weight']:.3f})")
    
    print(f"\nAdvanced features included:")
    print(f"  ✓ SentimentAspectIntegrator helper class")
    print(f"  ✓ Enhanced multilingual rule-based fallback")
    print(f"  ✓ Advanced batch processing with progress tracking")
    print(f"  ✓ Dynamic memory management")
    print(f"  ✓ Context-aware sentiment detection")
    
    print(f"\nTwo-model ensemble with advanced features ready!")
    
    # Cleanup
    classifier.cleanup()