#!/usr/bin/env python3
"""
Enhanced Sentiment Classifier - Two-Model Ensemble with Advanced Features
XLM-RoBERTa + Twitter-RoBERTa ensemble with restored advanced functionality
"""

from pyexpat import features
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
    XLM-RoBERTa (60.0%) + Twitter-RoBERTa (40.0%) + Advanced Features
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
        
        # Initialize vocabulary compilation tracking
        self._compiled_word_scores = False
        self._word_score_dict = {}
        self._phrase_score_dict = {}
        self._negation_set = set()
        self._intensifier_set = set()

        # Increase cache size for larger vocabulary coverage
        self.max_cache_size = 2000  # Increased from 1000

        # Pre-compile word lists on first use for better performance
        self._lazy_compile_enabled = True

    # Add to class methods:
    def _ensure_compiled_vocabularies(self):
        """Lazy compilation of vocabularies on first use"""
        if not self._compiled_word_scores and hasattr(self, '_lazy_compile_enabled'):
            word_lists = self._get_multilingual_word_lists()
            self._compile_word_score_lookups(word_lists)
            self._compiled_word_scores = True
        
        
    
    def _configure_device(self, device_preference: str) -> str:
        """Configure device with automatic fallback"""
        if device_preference == 'cuda':
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
        # Print header
        print("\n" + "="*70)
        print("ENHANCED SENTIMENT CLASSIFIER - TWO-MODEL ENSEMBLE")
        print("="*70)

        # Print device info
        print(f"Device: {self.device.upper()}")
        
        # Print GPU details if using CUDA
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"VRAM: {total_memory:.1f} GB")
            print("GPU acceleration enabled")
        else:
            print("Running on CPU")
        
        print("Models: cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual (60.0%) + cardiffnlp/twitter-roberta-base-sentiment-latest (40.0%)")
        print("="*70)
    
    def _load_two_model_ensemble(self):
        """Load the optimized two-model ensemble"""
        
        # Load environment variables for model configuration
        import os
        
        # Two-model configuration with environment variable support
        model_configs = [
            {
                'name': 'xlm_roberta',
                'model_id': os.environ.get('SENTIMENT_MODEL_1', 'cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual'),
                'fallback_id': 'distilbert-base-uncased-finetuned-sst-2-english',
                'weight': float(os.environ.get('SENTIMENT_MODEL_1_WEIGHT', '0.6')),  # 60% - primary model
                'max_length': 512
            },
            {
                'name': 'twitter_roberta',
                'model_id': os.environ.get('SENTIMENT_MODEL_2', 'cardiffnlp/twitter-roberta-base-sentiment-latest'),
                'fallback_id': 'distilbert-base-uncased-finetuned-sst-2-english', 
                'weight': float(os.environ.get('SENTIMENT_MODEL_2_WEIGHT', '0.4')),  # 40% - secondary model
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
            # Only FP32 strategies
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
    
    # Replace the existing analyze_sentiment method with this optimized version:

    def analyze_sentiment(self, text: str, language: str = 'auto') -> Dict:
        """
        Analyze sentiment with enhanced multilingual vocabularies
        """
        if not text or not text.strip():
            return self._neutral_result()
        
        # Ensure vocabularies are compiled for performance
        self._ensure_compiled_vocabularies()
        
        # Check cache first
        cache_key = hash(text[:100])
        if cache_key in self.cache:
            cached_result = self.cache[cache_key].copy()
            cached_result['from_cache'] = True
            return cached_result
        
        start_time = time.time()
        
        # Enhanced language detection
        detected_language = self._detect_language(text) if language == 'auto' else language
        
        # Get predictions from ensemble models
        predictions = self._get_model_predictions(text)
        
        if not predictions:
            # Use enhanced rule-based fallback with language context
            result = self._advanced_rule_based_analysis(text, detected_language)
            result['method'] = 'enhanced_rule_based'
            result['models_available'] = 0
        else:
            # Combine predictions from ensemble
            result = self._combine_predictions(predictions)
            result['method'] = 'two_model_ensemble'
            result['models_available'] = len(predictions)
        
        # Add enhanced metadata
        processing_time = time.time() - start_time
        result.update({
            'language': detected_language,
            'processing_time': processing_time,
            'device': self.device,
            'models_used': len(predictions),
            'from_cache': False,
            'vocabulary_version': 'enhanced_multilingual_v2',
            'language_confidence': self._calculate_language_confidence(text, detected_language)
        })
        
        # Ensure confidence is normalized (0-1 range)
        result['confidence'] = min(1.0, max(0.0, result['confidence']))
        
        # Cache result
        self._add_to_cache(cache_key, result)
        
        return result

    def _advanced_rule_based_analysis(self, text: str, detected_language: str = 'english') -> Dict:
        """Enhanced rule-based analysis with language context"""
        if not text or not text.strip():
            return self._neutral_result()
        
        text_lower = text.lower()
        
        # Analyze text features with language context
        text_features = self._analyze_text_features(text, text_lower)
        text_features['detected_language'] = detected_language
        
        # Get multilingual word lists (cached/pre-compiled)
        word_lists = self._get_multilingual_word_lists()
        
        # Calculate context-aware scores (optimized version)
        scores = self._calculate_context_aware_scores(text_lower, word_lists, text_features)
        
        # Apply language-specific dynamic thresholds
        thresholds = self._calculate_dynamic_thresholds(text_features)
        
        # Determine final sentiment
        return self._determine_adaptive_sentiment(scores, thresholds, text_features)

    def _calculate_language_confidence(self, text: str, detected_language: str) -> float:
        """Calculate confidence in language detection"""
        if detected_language in ['multilingual', 'unknown']:
            return 0.5
        
        # Simple confidence based on language-specific indicators found
        text_lower = text.lower().split()
        
        # This is a simplified version - could be enhanced further
        language_indicators = {
            'english': ['the', 'and', 'is', 'was', 'a', 'an'],
            'spanish': ['el', 'la', 'y', 'es', 'de', 'que'],
            'german': ['der', 'die', 'das', 'und', 'ist', 'war'],
            'french': ['le', 'la', 'et', 'est', 'de', 'que'],
            'dutch': ['de', 'het', 'en', 'is', 'van', 'dat']
        }
        
        indicators = language_indicators.get(detected_language, [])
        matches = sum(1 for word in text_lower if word in indicators)
        
        return min(0.9, 0.3 + (matches / max(len(text_lower), 1)) * 2)
    
    
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
    
    def _analyze_text_features(self, text: str, text_lower: str) -> Dict:
        """Analyze text characteristics for better sentiment detection in multiple languages"""
        multilingual_features = {
            'negations': {
                'english': [
                    'not', 'no', "doesn't", "won't", "can't", "never", "nothing", "none", "nowhere",
                    "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't",
                    "couldn't", "wouldn't", "shouldn't", "mustn't", "needn't", "daren't",
                    "shan't", "mightn't", "oughtn't", "without", "lacking", "absent", "missing",
                    "deny", "refuse", "reject", "decline", "forbid", "prohibit", "prevent",
                    "avoid", "cease", "stop", "quit", "fail", "unable", "impossible",
                    "neither", "nor", "hardly", "barely", "scarcely", "seldom", "rarely"
                ],
                'spanish': [
                    'no', 'nunca', 'nada', 'nadie', 'ningún', 'tampoco', 'jamás',
                    'ninguno', 'ninguna', 'ni', 'sin', 'carece', 'falta', 'ausente',
                    'rechazar', 'negar', 'denegar', 'prohibir', 'impedir', 'evitar',
                    'cesar', 'parar', 'fallar', 'incapaz', 'imposible', 'apenas',
                    'raramente', 'escasamente', 'difícilmente', 'negativo', 'negativamente',
                    'contraproducente', 'ineficaz', 'inútil', 'vano', 'fallido'
                ],
                'german': [
                    'nicht', 'keine', 'nein', 'niemals', 'niemand', 'nichts', 'kein',
                    'nirgends', 'nirgendwo', 'weder', 'noch', 'ohne', 'fehlt', 'abwesend',
                    'verweigern', 'ablehnen', 'verbieten', 'verhindern', 'vermeiden',
                    'aufhören', 'stoppen', 'versagen', 'unfähig', 'unmöglich', 'kaum',
                    'selten', 'knapp', 'schwer', 'negativ', 'kontraproduktiv',
                    'unwirksam', 'nutzlos', 'vergeblich', 'gescheitert', 'keinesfalls'
                ],
                'french': [
                    'ne', 'pas', 'jamais', 'rien', 'personne', 'aucun', 'ni',
                    'nulle', 'nul', 'point', 'sans', 'manque', 'absent', 'manquer',
                    'refuser', 'nier', 'rejeter', 'interdire', 'empêcher', 'éviter',
                    'cesser', 'arrêter', 'échouer', 'incapable', 'impossible', 'à peine',
                    'rarement', 'difficilement', 'négatif', 'négativement', 'inefficace',
                    'inutile', 'vain', 'raté', 'nullement', 'guère'
                ],
                'dutch': [
                    'niet', 'nee', 'nooit', 'niemand', 'niets', 'geen', 'nergens',
                    'zonder', 'ontbreekt', 'afwezig', 'weigeren', 'ontkennen', 'verbieden',
                    'voorkomen', 'vermijden', 'stoppen', 'ophouden', 'falen', 'onmogelijk',
                    'nauwelijks', 'zelden', 'schaars', 'moeilijk', 'negatief', 'nutteloos',
                    'tevergeefs', 'mislukt', 'geenszins', 'allerminst', 'weinig'
                ]
            },
            'intensifiers': {
                'english': [
                    'very', 'extremely', 'really', 'totally', 'completely', 'absolutely',
                    'incredibly', 'remarkably', 'exceptionally', 'extraordinarily', 'tremendously',
                    'enormously', 'immensely', 'vastly', 'hugely', 'massively', 'intensely',
                    'severely', 'deeply', 'profoundly', 'thoroughly', 'utterly', 'entirely',
                    'wholly', 'fully', 'perfectly', 'purely', 'genuinely', 'truly',
                    'seriously', 'highly', 'strongly', 'powerfully', 'dramatically',
                    'significantly', 'considerably', 'substantially', 'markedly', 'notably',
                    'particularly', 'especially', 'awfully', 'terribly', 'frighteningly'
                ],
                'spanish': [
                    'muy', 'extremadamente', 'realmente', 'totalmente', 'completamente', 'absolutamente',
                    'increíblemente', 'notablemente', 'excepcionalmente', 'extraordinariamente',
                    'tremendamente', 'enormemente', 'inmensamente', 'vastamente', 'intensamente',
                    'severamente', 'profundamente', 'completamente', 'enteramente', 'perfectamente',
                    'puramente', 'genuinamente', 'verdaderamente', 'seriamente', 'altamente',
                    'fuertemente', 'poderosamente', 'dramáticamente', 'significativamente',
                    'considerablemente', 'sustancialmente', 'marcadamente', 'particularmente',
                    'especialmente', 'terriblemente', 'espantosamente'
                ],
                'german': [
                    'sehr', 'äußerst', 'wirklich', 'total', 'komplett', 'absolut', 'völlig',
                    'unglaublich', 'bemerkenswert', 'außergewöhnlich', 'außerordentlich',
                    'enorm', 'immens', 'gewaltig', 'intensiv', 'schwer', 'tief',
                    'gründlich', 'vollständig', 'ganz', 'perfekt', 'rein', 'echt',
                    'ernsthaft', 'hochgradig', 'stark', 'kraftvoll', 'dramatisch',
                    'bedeutend', 'erheblich', 'wesentlich', 'merklich', 'besonders',
                    'schrecklich', 'furchtbar', 'unheimlich', 'wahnsinnig'
                ],
                'french': [
                    'très', 'extrêmement', 'vraiment', 'totalement', 'complètement', 'absolument',
                    'incroyablement', 'remarquablement', 'exceptionnellement', 'extraordinairement',
                    'énormément', 'immensément', 'intensément', 'sévèrement', 'profondément',
                    'complètement', 'entièrement', 'parfaitement', 'purement', 'véritablement',
                    'sérieusement', 'hautement', 'fortement', 'puissamment', 'dramatiquement',
                    'considérablement', 'substantiellement', 'particulièrement', 'spécialement',
                    'terriblement', 'affreusement', 'horriblement', 'formidablement'
                ],
                'dutch': [
                    'zeer', 'extreem', 'echt', 'totaal', 'compleet', 'absoluut', 'volledig',
                    'ongelooflijk', 'opmerkelijk', 'uitzonderlijk', 'buitengewoon',
                    'enorm', 'immens', 'geweldig', 'intens', 'zwaar', 'diep',
                    'grondig', 'geheel', 'perfect', 'puur', 'echt', 'serieus',
                    'sterk', 'krachtig', 'dramatisch', 'aanzienlijk', 'wezenlijk',
                    'merkbaar', 'bijzonder', 'vooral', 'verschrikkelijk', 'vreselijk',
                    'ontzettend', 'waanzinnig', 'razend'
                ]
            },
            'superlatives': {
                'english': [
                    'most', 'least', 'ever', 'always', 'never', 'best', 'worst',
                    'greatest', 'smallest', 'largest', 'highest', 'lowest', 'fastest',
                    'slowest', 'strongest', 'weakest', 'brightest', 'darkest',
                    'perfect', 'terrible', 'amazing', 'awful', 'fantastic', 'horrible',
                    'excellent', 'outstanding', 'supreme', 'ultimate', 'maximum',
                    'minimum', 'top', 'bottom', 'first', 'last', 'prime', 'peak',
                    'extreme', 'ultimate', 'unmatched', 'unparalleled', 'unprecedented',
                    'record-breaking', 'world-class', 'legendary', 'iconic'
                ],
                'spanish': [
                    'más', 'menos', 'siempre', 'nunca', 'mejor', 'peor', 'mayor',
                    'menor', 'máximo', 'mínimo', 'superior', 'inferior', 'óptimo',
                    'pésimo', 'excelente', 'terrible', 'increíble', 'horrible',
                    'fantástico', 'espantoso', 'perfecto', 'fatal', 'supremo',
                    'último', 'primero', 'principal', 'extremo', 'récord',
                    'sin igual', 'incomparable', 'inigualable', 'legendario', 'icónico'
                ],
                'german': [
                    'beste', 'schlechteste', 'meiste', 'wenigste', 'immer', 'nie',
                    'größte', 'kleinste', 'höchste', 'niedrigste', 'stärkste', 'schwächste',
                    'schnellste', 'langsamste', 'perfekt', 'schrecklich', 'ausgezeichnet',
                    'furchtbar', 'fantastisch', 'entsetzlich', 'hervorragend', 'gräßlich',
                    'überragend', 'minderwertig', 'erstklassig', 'letztklassig', 'spitze',
                    'extrem', 'ultimativ', 'unübertroffen', 'beispiellos', 'rekord',
                    'weltklasse', 'legendär', 'ikonisch'
                ],
                'french': [
                    'meilleur', 'pire', 'plus', 'moins', 'toujours', 'jamais',
                    'plus grand', 'plus petit', 'maximum', 'minimum', 'supérieur',
                    'inférieur', 'excellent', 'terrible', 'parfait', 'horrible',
                    'fantastique', 'épouvantable', 'remarquable', 'affreux',
                    'suprême', 'ultime', 'premier', 'dernier', 'extrême',
                    'inégalé', 'incomparable', 'sans précédent', 'record',
                    'légendaire', 'iconique', 'classe mondiale'
                ],
                'dutch': [
                    'beste', 'slechtste', 'meest', 'minst', 'altijd', 'nooit',
                    'grootste', 'kleinste', 'hoogste', 'laagste', 'sterkste', 'zwakste',
                    'snelste', 'langzaamste', 'perfect', 'verschrikkelijk', 'uitstekend',
                    'afschuwelijk', 'fantastisch', 'gruwelijk', 'voortreffelijk', 'vreselijk',
                    'superieur', 'inferieur', 'eersteklas', 'tweederangs', 'top',
                    'extreem', 'ultiem', 'ongeëvenaard', 'ongekend', 'record',
                    'wereldklasse', 'legendarisch', 'iconisch'
                ]
            },
            'profanity': {
                'english': [
                    'trash', 'garbage', 'crap', 'sucks', 'damn', 'shit', 'fuck', 'useless', 'terrible', 'horrible',
                    'awful', 'disgusting', 'revolting', 'appalling', 'abominable', 'atrocious',
                    'deplorable', 'despicable', 'detestable', 'loathsome', 'repugnant',
                    'vile', 'wretched', 'pathetic', 'worthless', 'hopeless', 'disaster',
                    'catastrophe', 'nightmare', 'hell', 'damn', 'bloody', 'bastard',
                    'stupid', 'idiotic', 'moronic', 'ridiculous', 'absurd', 'insane',
                    'crazy', 'mad', 'nuts', 'broken', 'busted', 'screwed', 'doomed',
                    'ruined', 'destroyed', 'annihilated', 'obliterated'
                ],
                'spanish': [
                    'basura', 'mierda', 'inútil', 'terrible', 'horrible', 'porquería', 'pésimo',
                    'asqueroso', 'repugnante', 'espantoso', 'abominable', 'atroz',
                    'deplorable', 'despreciable', 'detestable', 'odioso', 'repulsivo',
                    'vil', 'miserable', 'patético', 'sin valor', 'desesperanzador',
                    'desastre', 'catástrofe', 'pesadilla', 'infierno', 'maldito',
                    'estúpido', 'idiota', 'imbécil', 'ridículo', 'absurdo', 'loco',
                    'roto', 'jodido', 'arruinado', 'destruido'
                ],
                'german': [
                    'müll', 'scheiße', 'nutzlos', 'schrecklich', 'furchtbar', 'beschissen',
                    'ekelhaft', 'widerlich', 'entsetzlich', 'abscheulich', 'grauenhaft',
                    'verabscheuungswürdig', 'verwerflich', 'erbärmlich', 'wertlos',
                    'hoffnungslos', 'katastrophe', 'alptraum', 'hölle', 'verdammt',
                    'dumm', 'idiotisch', 'schwachsinnig', 'lächerlich', 'absurd',
                    'verrückt', 'wahnsinnig', 'kaputt', 'im eimer', 'ruiniert',
                    'zerstört', 'vernichtet'
                ],
                'french': [
                    'merde', 'ordure', 'inutile', 'terrible', 'horrible', 'pourri', 'nul',
                    'dégoûtant', 'répugnant', 'épouvantable', 'abominable', 'atroce',
                    'déplorable', 'méprisable', 'détestable', 'odieux', 'répulsif',
                    'vil', 'misérable', 'pathétique', 'sans valeur', 'désespéré',
                    'désastre', 'catastrophe', 'cauchemar', 'enfer', 'maudit',
                    'stupide', 'idiot', 'imbécile', 'ridicule', 'absurde', 'fou',
                    'cassé', 'foutu', 'ruiné', 'détruit', 'anéanti'
                ],
                'dutch': [
                    'rotzooi', 'shit', 'waardeloos', 'verschrikkelijk', 'klote', 'kut',
                    'walgelijk', 'weerzinwekkend', 'afschuwelijk', 'gruwelijk', 'vreselijk',
                    'verachtelijk', 'verafschuwelijk', 'ellendig', 'nutteloos',
                    'hopeloos', 'ramp', 'catastrofe', 'nachtmerrie', 'hel', 'verdomd',
                    'dom', 'idioot', 'debiel', 'belachelijk', 'absurd', 'gek',
                    'kapot', 'naar de klote', 'verneukt', 'geruïneerd', 'vernietigd'
                ]
            },
            'emotion_markers': {
                'english': [
                    'excited', 'thrilled', 'delighted', 'overjoyed', 'ecstatic', 'elated',
                    'frustrated', 'annoyed', 'irritated', 'furious', 'enraged', 'livid',
                    'disappointed', 'heartbroken', 'devastated', 'crushed', 'shocked',
                    'surprised', 'amazed', 'astonished', 'confused', 'puzzled', 'worried',
                    'anxious', 'nervous', 'scared', 'terrified', 'relieved', 'grateful'
                ],
                'spanish': [
                    'emocionado', 'encantado', 'eufórico', 'extasiado', 'exaltado',
                    'frustrado', 'molesto', 'irritado', 'furioso', 'enfurecido', 'livido',
                    'decepcionado', 'desconsolado', 'devastado', 'aplastado', 'conmocionado',
                    'sorprendido', 'asombrado', 'pasmado', 'confundido', 'perplejo',
                    'preocupado', 'ansioso', 'nervioso', 'asustado', 'aterrorizado',
                    'aliviado', 'agradecido'
                ],
                'german': [
                    'aufgeregt', 'begeistert', 'erfreut', 'überglücklich', 'ekstatisch',
                    'frustriert', 'verärgert', 'irritiert', 'wütend', 'rasend', 'zornig',
                    'enttäuscht', 'am boden zerstört', 'vernichtet', 'geschockt',
                    'überrascht', 'erstaunt', 'verblüfft', 'verwirrt', 'ratlos',
                    'besorgt', 'ängstlich', 'nervös', 'verängstigt', 'terrorisiert',
                    'erleichtert', 'dankbar'
                ],
                'french': [
                    'excité', 'ravi', 'enchanté', 'fou de joie', 'extatique', 'exalté',
                    'frustré', 'agacé', 'irrité', 'furieux', 'enragé', 'livide',
                    'déçu', 'le cœur brisé', 'dévasté', 'écrasé', 'choqué',
                    'surpris', 'étonné', 'stupéfait', 'confus', 'perplexe',
                    'inquiet', 'anxieux', 'nerveux', 'effrayé', 'terrorisé',
                    'soulagé', 'reconnaissant'
                ],
                'dutch': [
                    'opgewonden', 'verrukt', 'verheugd', 'dolblij', 'extatisch',
                    'gefrustreerd', 'geërgerd', 'geïrriteerd', 'woedend', 'razend',
                    'teleurgesteld', 'gebroken', 'verwoest', 'verbrijzeld', 'geschokt',
                    'verrast', 'verbaasd', 'stomverbaasd', 'verward', 'puzzled',
                    'bezorgd', 'angstig', 'nerveus', 'bang', 'doodsbang',
                    'opgelucht', 'dankbaar'
                ]
            }
        }

        # Flatten word lists for each feature (including new emotion markers)
        all_negations = [word for lang_list in multilingual_features['negations'].values() for word in lang_list]
        all_intensifiers = [word for lang_list in multilingual_features['intensifiers'].values() for word in lang_list]
        all_superlatives = [word for lang_list in multilingual_features['superlatives'].values() for word in lang_list]
        all_profanity = [word for lang_list in multilingual_features['profanity'].values() for word in lang_list]
        all_emotion_markers = [word for lang_list in multilingual_features['emotion_markers'].values() for word in lang_list]

        features = {
            'length': len(text.split()),
            'has_caps': bool(re.search(r'[A-Z]{3,}', text)),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'has_negations': any(neg in text_lower for neg in all_negations),
            'has_intensifiers': any(intens in text_lower for intens in all_intensifiers),
            'has_superlatives': any(sup in text_lower for sup in all_superlatives),
            'has_emotion_markers': any(emotion in text_lower for emotion in all_emotion_markers),
            'punctuation_emphasis': text.count('!') + text.count('?') + len(re.findall(r'[.]{2,}', text)),
            'has_profanity': any(prof in text_lower for prof in all_profanity),
            'original_text': text_lower,
            
            # Additional linguistic features
            'repetition_count': len(re.findall(r'(.)\1{2,}', text_lower)),  # repeated characters
            'caps_ratio': len(re.findall(r'[A-Z]', text)) / max(len(text), 1),
            'punctuation_density': len(re.findall(r'[!?.,;:]', text)) / max(len(text.split()), 1),
            'negation_intensity': sum(1 for neg in all_negations if neg in text_lower),
            'intensifier_count': sum(1 for intens in all_intensifiers if intens in text_lower),
            'superlative_count': sum(1 for sup in all_superlatives if sup in text_lower),
            'emotion_word_count': sum(1 for emotion in all_emotion_markers if emotion in text_lower)
        }

        # Calculate enhanced emphasis level with multilingual support
        features['emphasis_level'] = min(1.0, (
            features['exclamation_count'] * 0.25 + 
            features['has_caps'] * 0.3 + 
            features['punctuation_emphasis'] * 0.15 +
            features['has_superlatives'] * 0.1 +
            features['caps_ratio'] * 0.1 +
            features['repetition_count'] * 0.05 +
            features['emotion_word_count'] * 0.05
        ))
        
        # Calculate sentiment polarity hints
        features['polarity_hints'] = {
            'negative_signals': features['negation_intensity'] + features['has_profanity'] * 2,
            'positive_signals': features['intensifier_count'] + features['superlative_count'],
            'emotional_intensity': features['emotion_word_count'] + features['emphasis_level']
        }

        return features
        
    
    def _get_multilingual_word_lists(self) -> Dict:
        """Enhanced multilingual word lists for better global coverage"""
        return {
            'strong_positive': {
                'english': [
                    'excellent', 'amazing', 'fantastic', 'perfect', 'love', 'best', 'awesome', 'outstanding', 
                    'brilliant', 'superb', 'wonderful', 'magnificent', 'spectacular', 'phenomenal', 'exceptional',
                    'extraordinary', 'marvelous', 'fabulous', 'incredible', 'unbelievable', 'breathtaking',
                    'stunning', 'remarkable', 'flawless', 'impeccable', 'sublime', 'divine', 'heavenly',
                    'blissful', 'ecstatic', 'euphoric', 'thrilled', 'delighted', 'overjoyed', 'elated',
                    'triumphant', 'victorious', 'glorious', 'legendary', 'masterpiece', 'genius', 'revolutionary',
                    'groundbreaking', 'life-changing', 'mind-blowing', 'world-class', 'top-tier', 'premium',
                    'superior', 'first-class', 'five-star', 'gold-standard', 'benchmark'
                ],
                'spanish': [
                    'excelente', 'increíble', 'fantástico', 'perfecto', 'amo', 'mejor', 'genial', 'maravilloso',
                    'brillante', 'magnífico', 'espectacular', 'fenomenal', 'excepcional', 'extraordinario',
                    'fabuloso', 'impresionante', 'asombroso', 'sorprendente', 'impecable', 'sublime',
                    'divino', 'celestial', 'dichoso', 'extático', 'eufórico', 'encantado', 'emocionado',
                    'radiante', 'triunfante', 'victorioso', 'glorioso', 'legendario', 'obra maestra',
                    'genio', 'revolucionario', 'innovador', 'transformador', 'alucinante', 'de primera',
                    'superior', 'premium', 'cinco estrellas', 'referencia', 'estándar de oro',
                    'insuperable', 'incomparable', 'sin igual', 'óptimo'
                ],
                'german': [
                    'ausgezeichnet', 'erstaunlich', 'fantastisch', 'perfekt', 'liebe', 'beste', 'toll', 'wunderbar',
                    'brillant', 'großartig', 'spektakulär', 'phänomenal', 'außergewöhnlich', 'extraordinär',
                    'fabelhaft', 'unglaublich', 'atemberaubend', 'überwältigend', 'tadellos', 'erhaben',
                    'göttlich', 'himmlisch', 'selig', 'ekstatisch', 'euphorisch', 'begeistert', 'entzückt',
                    'triumphierend', 'siegreich', 'glorreich', 'legendär', 'meisterwerk', 'genie',
                    'revolutionär', 'bahnbrechend', 'lebensverändernd', 'umwerfend', 'weltklasse',
                    'erstklassig', 'premium', 'überlegen', 'fünf-sterne', 'goldstandard', 'maßstab',
                    'unschlagbar', 'unvergleichlich', 'einzigartig', 'optimal'
                ],
                'french': [
                    'excellent', 'incroyable', 'fantastique', 'parfait', 'amour', 'meilleur', 'génial', 'merveilleux',
                    'brillant', 'magnifique', 'spectaculaire', 'phénoménal', 'exceptionnel', 'extraordinaire',
                    'fabuleux', 'impressionnant', 'époustouflant', 'saisissant', 'impeccable', 'sublime',
                    'divin', 'céleste', 'béni', 'extatique', 'euphorique', 'ravi', 'enchanté',
                    'triomphant', 'victorieux', 'glorieux', 'légendaire', 'chef-d\'œuvre', 'génie',
                    'révolutionnaire', 'révolutionnaire', 'transformateur', 'époustouflant', 'classe mondiale',
                    'première classe', 'premium', 'supérieur', 'cinq étoiles', 'étalon-or', 'référence',
                    'imbattable', 'incomparable', 'unique', 'optimal'
                ],
                'dutch': [
                    'uitstekend', 'geweldig', 'fantastisch', 'perfect', 'houd van', 'beste', 'geweldig',
                    'briljant', 'prachtig', 'spectaculair', 'fenomenaal', 'uitzonderlijk', 'buitengewoon',
                    'fabelachtig', 'ongelooflijk', 'adembenemend', 'overweldigend', 'onberispelijk', 'verheven',
                    'goddelijk', 'hemels', 'zalig', 'extatisch', 'euforisch', 'verrukt', 'opgetogen',
                    'triomfantelijk', 'zegevierend', 'glorieus', 'legendarisch', 'meesterwerk', 'genie',
                    'revolutionair', 'baanbrekend', 'levensveranderend', 'verbijsterend', 'wereldklasse',
                    'eersteklas', 'premium', 'superieur', 'vijf-sterren', 'goudstandaard', 'benchmark',
                    'onovertroffen', 'onvergelijkbaar', 'uniek', 'optimaal'
                ]
            },
            'strong_negative': {
                'english': [
                    'terrible', 'horrible', 'awful', 'worst', 'hate', 'useless', 'garbage', 'trash', 'disaster', 
                    'nightmare', 'broken', 'trashiest', 'laziest', 'slowest', 'disgusting', 'revolting',
                    'appalling', 'abominable', 'atrocious', 'deplorable', 'despicable', 'detestable',
                    'loathsome', 'repugnant', 'vile', 'wretched', 'pathetic', 'worthless', 'hopeless',
                    'catastrophic', 'devastating', 'ruinous', 'destructive', 'disastrous', 'calamitous',
                    'tragic', 'dreadful', 'ghastly', 'hideous', 'monstrous', 'outrageous', 'scandalous',
                    'shameful', 'disgraceful', 'contemptible', 'abhorrent', 'odious', 'heinous',
                    'execrable', 'insufferable', 'intolerable', 'unbearable', 'unacceptable'
                ],
                'spanish': [
                    'terrible', 'horrible', 'espantoso', 'peor', 'odio', 'inútil', 'basura', 'desastre',
                    'pesadilla', 'roto', 'asqueroso', 'repugnante', 'atroz', 'abominable', 'deplorable',
                    'despreciable', 'detestable', 'odioso', 'vil', 'miserable', 'patético', 'sin valor',
                    'catastrófico', 'devastador', 'ruinoso', 'destructivo', 'desastroso', 'calamitoso',
                    'trágico', 'espantoso', 'horroroso', 'monstruoso', 'escandaloso', 'vergonzoso',
                    'deshonroso', 'despreciable', 'aborrecible', 'execrable', 'insufrible', 'intolerable',
                    'insoportable', 'inaceptable', 'imperdonable', 'lamentable', 'pésimo', 'fatal',
                    'nefasto', 'funesto', 'siniestro'
                ],
                'german': [
                    'schrecklich', 'furchtbar', 'schlimm', 'schlechteste', 'hasse', 'nutzlos', 'müll', 'katastrophe',
                    'alptraum', 'kaputt', 'widerlich', 'ekelhaft', 'grauenhaft', 'abscheulich', 'verwerflich',
                    'verachtenswert', 'verabscheuungswürdig', 'widerlich', 'elend', 'erbärmlich', 'wertlos',
                    'katastrophal', 'verheerend', 'ruinös', 'zerstörerisch', 'desaströs', 'verhängnisvoll',
                    'tragisch', 'grässlich', 'scheußlich', 'monströs', 'empörend', 'skandalös',
                    'schändlich', 'ehrlos', 'verachtenswert', 'abscheulich', 'verwerflich', 'unerträglich',
                    'untragbar', 'inakzeptabel', 'unverzeihlich', 'beklagenswert', 'miserabel',
                    'verhängnisvoll', 'unheilbringend', 'finster'
                ],
                'french': [
                    'terrible', 'horrible', 'affreux', 'pire', 'déteste', 'inutile', 'ordures', 'désastre',
                    'cauchemar', 'cassé', 'dégoûtant', 'répugnant', 'atroce', 'abominable', 'déplorable',
                    'méprisable', 'détestable', 'odieux', 'vil', 'misérable', 'pathétique', 'sans valeur',
                    'catastrophique', 'dévastateur', 'ruineux', 'destructeur', 'désastreux', 'calamiteux',
                    'tragique', 'épouvantable', 'hideux', 'monstrueux', 'scandaleux', 'honteux',
                    'déshonorant', 'méprisable', 'abhorrent', 'exécrable', 'insupportable', 'intolérable',
                    'inadmissible', 'impardonnable', 'lamentable', 'lamentable', 'nul',
                    'néfaste', 'funeste', 'sinistre'
                ],
                'dutch': [
                    'verschrikkelijk', 'afschuwelijk', 'vreselijk', 'ergste', 'haat', 'nutteloos', 'afval',
                    'ramp', 'nachtmerrie', 'kapot', 'walgelijk', 'weerzinwekkend', 'gruwelijk', 'abominabel',
                    'deplorabele', 'verachtelijk', 'afschuwelijk', 'gemeen', 'ellendig', 'zielig', 'waardeloos',
                    'catastrofaal', 'verwoestend', 'ruïneus', 'destructief', 'rampzalig', 'rampspoed',
                    'tragisch', 'griezelig', 'afzichtelijk', 'monsterachtig', 'schandalig', 'beschamend',
                    'schandelijk', 'verachtelijk', 'afkeurenswaardig', 'ondraaglijk', 'onverdragelijk',
                    'onaanvaardbaar', 'onvergeeflijk', 'beklagenswaardig', 'beroerd', 'waardeloos',
                    'noodlottig', 'onheilspellend', 'sinister'
                ]
            },
            'moderate_positive': {
                'english': [
                    'good', 'great', 'nice', 'like', 'helpful', 'useful', 'works', 'satisfied', 'happy', 'recommend',
                    'pleasant', 'enjoyable', 'decent', 'solid', 'reliable', 'effective', 'efficient', 'smooth',
                    'comfortable', 'convenient', 'easy', 'simple', 'clear', 'clean', 'fresh', 'bright',
                    'positive', 'optimistic', 'cheerful', 'friendly', 'warm', 'welcoming', 'inviting',
                    'appealing', 'attractive', 'charming', 'delightful', 'pleasing', 'agreeable', 'favorable',
                    'beneficial', 'valuable', 'worthwhile', 'rewarding', 'fulfilling', 'satisfying', 'gratifying',
                    'encouraging', 'promising', 'hopeful', 'successful', 'productive', 'constructive'
                ],
                'spanish': [
                    'bueno', 'genial', 'agradable', 'gusta', 'útil', 'funciona', 'satisfecho', 'feliz',
                    'placentero', 'disfrutable', 'decente', 'sólido', 'confiable', 'efectivo', 'eficiente',
                    'suave', 'cómodo', 'conveniente', 'fácil', 'simple', 'claro', 'limpio', 'fresco',
                    'brillante', 'positivo', 'optimista', 'alegre', 'amigable', 'cálido', 'acogedor',
                    'atractivo', 'encantador', 'delicioso', 'agradable', 'favorable', 'beneficioso',
                    'valioso', 'gratificante', 'satisfactorio', 'alentador', 'prometedor', 'esperanzador',
                    'exitoso', 'productivo', 'constructivo', 'provechoso', 'ventajoso'
                ],
                'german': [
                    'gut', 'toll', 'schön', 'mag', 'hilfreich', 'nützlich', 'funktioniert', 'zufrieden',
                    'angenehm', 'erfreulich', 'ordentlich', 'solide', 'zuverlässig', 'effektiv', 'effizient',
                    'glatt', 'bequem', 'praktisch', 'einfach', 'simpel', 'klar', 'sauber', 'frisch',
                    'hell', 'positiv', 'optimistisch', 'fröhlich', 'freundlich', 'warm', 'einladend',
                    'ansprechend', 'attraktiv', 'charmant', 'erfreulich', 'gefällig', 'günstig',
                    'vorteilhaft', 'wertvoll', 'lohnend', 'erfüllend', 'befriedigend', 'ermutigend',
                    'vielversprechend', 'hoffnungsvoll', 'erfolgreich', 'produktiv', 'konstruktiv'
                ],
                'french': [
                    'bon', 'super', 'agréable', 'aime', 'utile', 'marche', 'satisfait', 'heureux',
                    'plaisant', 'agréable', 'décent', 'solide', 'fiable', 'efficace', 'efficient',
                    'lisse', 'confortable', 'pratique', 'facile', 'simple', 'clair', 'propre', 'frais',
                    'lumineux', 'positif', 'optimiste', 'joyeux', 'amical', 'chaleureux', 'accueillant',
                    'séduisant', 'attrayant', 'charmant', 'délicieux', 'plaisant', 'favorable',
                    'bénéfique', 'précieux', 'gratifiant', 'satisfaisant', 'encourageant', 'prometteur',
                    'plein d\'espoir', 'réussi', 'productif', 'constructif', 'avantageux'
                ],
                'dutch': [
                    'goed', 'geweldig', 'leuk', 'vind leuk', 'handig', 'werkt', 'tevreden', 'blij',
                    'aangenaam', 'plezierig', 'degelijk', 'solide', 'betrouwbaar', 'effectief', 'efficiënt',
                    'soepel', 'comfortabel', 'handig', 'makkelijk', 'eenvoudig', 'helder', 'schoon', 'fris',
                    'helder', 'positief', 'optimistisch', 'vrolijk', 'vriendelijk', 'warm', 'uitnodigend',
                    'aantrekkelijk', 'charmant', 'heerlijk', 'aangenaam', 'gunstig', 'voordelig',
                    'waardevol', 'de moeite waard', 'bevredigend', 'bemoedigend', 'veelbelovend',
                    'hoopvol', 'succesvol', 'productief', 'constructief', 'voordelig'
                ]
            },
            'moderate_negative': {
                'english': [
                    'bad', 'poor', 'disappointing', 'frustrated', 'annoying', 'slow', 'difficult', 'confusing', 
                    'problem', 'issue', 'bug', 'crash', 'sucks', 'mediocre', 'average', 'subpar', 'inferior',
                    'lacking', 'insufficient', 'inadequate', 'unsatisfactory', 'unimpressive', 'underwhelming',
                    'boring', 'dull', 'tedious', 'tiresome', 'mundane', 'repetitive', 'monotonous',
                    'uncomfortable', 'inconvenient', 'clunky', 'awkward', 'cumbersome', 'complicated',
                    'unclear', 'messy', 'disorganized', 'chaotic', 'unreliable', 'unstable', 'inconsistent',
                    'flawed', 'faulty', 'defective', 'damaged', 'worn', 'outdated', 'obsolete'
                ],
                'spanish': [
                    'malo', 'pobre', 'decepcionante', 'frustrado', 'molesto', 'lento', 'difícil', 'problema',
                    'confuso', 'falla', 'fallo', 'mediocre', 'promedio', 'inferior', 'deficiente',
                    'insuficiente', 'inadecuado', 'insatisfactorio', 'poco impresionante', 'decepcionante',
                    'aburrido', 'soso', 'tedioso', 'cansado', 'mundano', 'repetitivo', 'monótono',
                    'incómodo', 'inconveniente', 'torpe', 'complicado', 'poco claro', 'desordenado',
                    'caótico', 'no confiable', 'inestable', 'inconsistente', 'defectuoso', 'dañado',
                    'gastado', 'desactualizado', 'obsoleto', 'problemático', 'irregular'
                ],
                'german': [
                    'schlecht', 'arm', 'enttäuschend', 'frustriert', 'ärgerlich', 'langsam', 'schwierig', 'problem',
                    'verwirrend', 'fehler', 'absturz', 'mittelmäßig', 'durchschnittlich', 'minderwertig',
                    'mangelhaft', 'unzureichend', 'unzulänglich', 'unbefriedigend', 'unbeeindruckend',
                    'langweilig', 'stumpf', 'ermüdend', 'lästig', 'alltäglich', 'sich wiederholend',
                    'eintönig', 'unbequem', 'unpraktisch', 'umständlich', 'kompliziert', 'unklar',
                    'unordentlich', 'chaotisch', 'unzuverlässig', 'instabil', 'widersprüchlich',
                    'fehlerhaft', 'defekt', 'beschädigt', 'abgenutzt', 'veraltet', 'überholt'
                ],
                'french': [
                    'mauvais', 'pauvre', 'décevant', 'frustré', 'ennuyeux', 'lent', 'difficile', 'problème',
                    'confus', 'bug', 'plantage', 'médiocre', 'moyen', 'inférieur', 'déficient',
                    'insuffisant', 'inadéquat', 'insatisfaisant', 'peu impressionnant', 'décevant',
                    'ennuyeux', 'terne', 'fastidieux', 'fatigant', 'banal', 'répétitif', 'monotone',
                    'inconfortable', 'peu pratique', 'maladroit', 'compliqué', 'peu clair', 'désordonné',
                    'chaotique', 'peu fiable', 'instable', 'incohérent', 'défectueux', 'endommagé',
                    'usé', 'dépassé', 'obsolète', 'problématique', 'irrégulier'
                ],
                'dutch': [
                    'slecht', 'arm', 'teleurstellend', 'gefrustreerd', 'vervelend', 'langzaam', 'moeilijk', 'probleem',
                    'verwarrend', 'bug', 'crash', 'matig', 'gemiddeld', 'inferieur', 'gebrekkig',
                    'onvoldoende', 'inadequaat', 'onbevredigend', 'niet indrukwekkend', 'teleurstellend',
                    'saai', 'duf', 'vermoeiend', 'vervelend', 'alledaags', 'repetitief', 'monotoon',
                    'oncomfortabel', 'onhandig', 'omslachtig', 'ingewikkeld', 'onduidelijk', 'rommelig',
                    'chaotisch', 'onbetrouwbaar', 'onstabiel', 'inconsistent', 'gebrekkig', 'defect',
                    'beschadigd', 'versleten', 'verouderd', 'achterhaald', 'problematisch'
                ]
            },
            'intensifiers': {
                'english': [
                    'very', 'extremely', 'really', 'totally', 'completely', 'absolutely', 'incredibly',
                    'remarkably', 'exceptionally', 'extraordinarily', 'tremendously', 'enormously',
                    'immensely', 'vastly', 'hugely', 'massively', 'intensely', 'severely', 'deeply',
                    'profoundly', 'thoroughly', 'utterly', 'entirely', 'wholly', 'fully', 'perfectly',
                    'purely', 'genuinely', 'truly', 'seriously', 'highly', 'strongly', 'powerfully',
                    'dramatically', 'significantly', 'considerably', 'substantially', 'markedly'
                ],
                'spanish': [
                    'muy', 'extremadamente', 'realmente', 'totalmente', 'completamente', 'absolutamente',
                    'increíblemente', 'notablemente', 'excepcionalmente', 'extraordinariamente',
                    'tremendamente', 'enormemente', 'inmensamente', 'vastamente', 'intensamente',
                    'severamente', 'profundamente', 'completamente', 'enteramente', 'perfectamente',
                    'puramente', 'genuinamente', 'verdaderamente', 'seriamente', 'altamente',
                    'fuertemente', 'poderosamente', 'dramáticamente', 'significativamente',
                    'considerablemente', 'sustancialmente', 'marcadamente'
                ],
                'german': [
                    'sehr', 'äußerst', 'wirklich', 'total', 'komplett', 'absolut', 'völlig',
                    'unglaublich', 'bemerkenswert', 'außergewöhnlich', 'außerordentlich', 'enorm',
                    'immens', 'gewaltig', 'intensiv', 'schwer', 'tief', 'gründlich', 'vollständig',
                    'ganz', 'perfekt', 'rein', 'echt', 'ernsthaft', 'hochgradig', 'stark',
                    'kraftvoll', 'dramatisch', 'bedeutend', 'erheblich', 'wesentlich', 'merklich'
                ],
                'french': [
                    'très', 'extrêmement', 'vraiment', 'totalement', 'complètement', 'absolument',
                    'incroyablement', 'remarquablement', 'exceptionnellement', 'extraordinairement',
                    'énormément', 'immensément', 'intensément', 'sévèrement', 'profondément',
                    'complètement', 'entièrement', 'parfaitement', 'purement', 'véritablement',
                    'sérieusement', 'hautement', 'fortement', 'puissamment', 'dramatiquement',
                    'considérablement', 'substantiellement', 'particulièrement'
                ],
                'dutch': [
                    'zeer', 'extreem', 'echt', 'totaal', 'compleet', 'absoluut', 'volledig',
                    'ongelooflijk', 'opmerkelijk', 'uitzonderlijk', 'buitengewoon', 'enorm',
                    'immens', 'geweldig', 'intens', 'zwaar', 'diep', 'grondig', 'geheel',
                    'perfect', 'puur', 'echt', 'serieus', 'sterk', 'krachtig', 'dramatisch',
                    'aanzienlijk', 'wezenlijk', 'merkbaar', 'bijzonder'
                ]
            },
            'negations': {
                'english': [
                    'not', 'no', "doesn't", "won't", "can't", "isn't", "never", 'nothing', 'nowhere', 'nobody',
                    "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "couldn't", "wouldn't",
                    "shouldn't", "mustn't", "needn't", "daren't", "shan't", "mightn't", "oughtn't",
                    'without', 'lacking', 'absent', 'missing', 'deny', 'refuse', 'reject', 'decline',
                    'forbid', 'prohibit', 'prevent', 'avoid', 'cease', 'stop', 'quit', 'fail',
                    'unable', 'impossible', 'neither', 'nor', 'hardly', 'barely', 'scarcely'
                ],
                'spanish': [
                    'no', 'nunca', 'nada', 'nadie', 'ningún', 'tampoco', 'jamás', 'ninguno', 'ninguna',
                    'ni', 'sin', 'carece', 'falta', 'ausente', 'rechazar', 'negar', 'denegar',
                    'prohibir', 'impedir', 'evitar', 'cesar', 'parar', 'fallar', 'incapaz',
                    'imposible', 'apenas', 'raramente', 'escasamente', 'difícilmente'
                ],
                'german': [
                    'nicht', 'keine', 'nein', 'niemals', 'niemand', 'nichts', 'kein', 'nirgends',
                    'nirgendwo', 'weder', 'noch', 'ohne', 'fehlt', 'abwesend', 'verweigern',
                    'ablehnen', 'verbieten', 'verhindern', 'vermeiden', 'aufhören', 'stoppen',
                    'versagen', 'unfähig', 'unmöglich', 'kaum', 'selten', 'knapp'
                ],
                'french': [
                    'ne', 'pas', 'jamais', 'rien', 'personne', 'aucun', 'ni', 'nulle', 'nul',
                    'point', 'sans', 'manque', 'absent', 'manquer', 'refuser', 'nier', 'rejeter',
                    'interdire', 'empêcher', 'éviter', 'cesser', 'arrêter', 'échouer', 'incapable',
                    'impossible', 'à peine', 'rarement', 'difficilement'
                ],
                'dutch': [
                    'niet', 'nee', 'nooit', 'niemand', 'niets', 'geen', 'nergens', 'zonder',
                    'ontbreekt', 'afwezig', 'weigeren', 'ontkennen', 'verbieden', 'voorkomen',
                    'vermijden', 'stoppen', 'ophouden', 'falen', 'onmogelijk', 'nauwelijks',
                    'zelden', 'schaars'
                ]
            }
        }
    
    def _calculate_context_aware_scores(self, text_lower: str, word_lists: Dict, features: Dict) -> Dict:
        """Optimized sentiment scores for large multilingual vocabularies"""
        scores = {'positive': 0.0, 'negative': 0.0}
        words = text_lower.split()
        
        # Pre-compile flat lookup dictionaries for faster word matching
        if not hasattr(self, '_compiled_word_scores'):
            self._compile_word_score_lookups(word_lists)
        
        for i, word in enumerate(words):
            # Get context window
            context_start = max(0, i - 3)
            context_end = min(len(words), i + 4)
            context = words[context_start:context_end]
            
            # Fast context checks using pre-compiled sets
            has_negation = bool(self._negation_set.intersection(context))
            has_intensifier = bool(self._intensifier_set.intersection(context))
            
            # Fast word score lookup
            word_score = self._word_score_dict.get(word, 0.0)
            
            # Check for phrase matches if single word not found
            if word_score == 0.0:
                phrase = ' '.join(words[max(0, i-1):i+2])
                word_score = self._phrase_score_dict.get(phrase, 0.0)
            
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
        
        # Normalize and apply emphasis
        text_words = max(len(words), 1)
        emphasis_multiplier = 1.0 + (features['emphasis_level'] * 0.3)
        
        scores['positive'] = (scores['positive'] / text_words) * emphasis_multiplier
        scores['negative'] = (scores['negative'] / text_words) * emphasis_multiplier
        
        return scores
        
    def _compile_word_score_lookups(self, word_lists: Dict):
        """Pre-compile word lookups for faster processing"""
        self._word_score_dict = {}
        self._phrase_score_dict = {}
        self._negation_set = set()
        self._intensifier_set = set()
        
        # Compile sentiment word scores
        for sentiment_type, lang_dict in word_lists.items():
            if sentiment_type == 'negations':
                for lang, words in lang_dict.items():
                    self._negation_set.update(words)
            elif sentiment_type == 'intensifiers':
                for lang, words in lang_dict.items():
                    self._intensifier_set.update(words)
            else:
                # Score assignment
                if 'strong' in sentiment_type:
                    score = 2.0 if 'positive' in sentiment_type else -2.0
                else:
                    score = 1.0 if 'positive' in sentiment_type else -1.0
                
                # Add all words/phrases to lookup
                for lang, word_list in lang_dict.items():
                    for word_or_phrase in word_list:
                        if len(word_or_phrase.split()) == 1:
                            self._word_score_dict[word_or_phrase] = score
                        else:
                            self._phrase_score_dict[word_or_phrase] = score    
        
            
    def _calculate_dynamic_thresholds(self, features: Dict) -> Dict:
        """Calculate dynamic thresholds with language-specific adjustments"""
        base_threshold = 0.8
        threshold_adjustments = 0.0
        
        # Get detected language
        language = features.get('detected_language', 'english')
        
        # Language-specific baseline adjustments
        language_adjustments = {
            'german': -0.05,    # German tends to be more direct/harsh
            'dutch': -0.03,     # Similar to German
            'spanish': 0.02,    # Often more expressive
            'french': 0.01,     # Slightly more formal
            'english': 0.0      # Baseline
        }
        
        threshold_adjustments += language_adjustments.get(language, 0.0)
        
        # Text length adjustments
        if features['length'] < 10:
            threshold_adjustments -= 0.3
        elif features['length'] > 30:
            threshold_adjustments += 0.1
        
        # High emphasis texts are easier to classify
        if features['emphasis_level'] > 0.5:
            threshold_adjustments -= 0.25
        
        # Profanity is a strong negative indicator
        if features['has_profanity']:
            # Language-specific profanity weights
            profanity_weights = {
                'german': -0.8,     # German profanity very strong
                'dutch': -0.75,
                'english': -0.75,
                'spanish': -0.7,
                'french': -0.65
            }
            threshold_adjustments += profanity_weights.get(language, -0.75)
        
        # Multiple negations compound the effect
        if features.get('negation_intensity', 0) > 2:
            threshold_adjustments -= 0.2
        
        # High emotion word count increases confidence
        if features.get('emotion_word_count', 0) > 2:
            threshold_adjustments -= 0.15
        
        positive_threshold = max(0.5, base_threshold + threshold_adjustments)
        negative_threshold = max(0.4, base_threshold + threshold_adjustments)
        
        return {
            'positive_threshold': positive_threshold,
            'negative_threshold': negative_threshold,
            'language_adjustment': language_adjustments.get(language, 0.0)
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
        """Enhanced language detection using vocabulary fingerprinting"""
        if not text or len(text.strip()) < 3:
            return 'unknown'
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Language fingerprint words (high-confidence indicators)
        language_indicators = {
            'english': {
                'articles': ['the', 'a', 'an'],
                'pronouns': ['i', 'you', 'he', 'she', 'it', 'we', 'they'],
                'prepositions': ['of', 'in', 'to', 'for', 'with', 'on', 'by'],
                'common_words': ['and', 'or', 'but', 'that', 'this', 'is', 'was', 'are', 'were']
            },
            'spanish': {
                'articles': ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas'],
                'pronouns': ['yo', 'tú', 'él', 'ella', 'nosotros', 'vosotros', 'ellos'],
                'prepositions': ['de', 'en', 'a', 'por', 'para', 'con', 'sin'],
                'common_words': ['y', 'o', 'pero', 'que', 'este', 'es', 'era', 'son', 'fueron']
            },
            'german': {
                'articles': ['der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einen'],
                'pronouns': ['ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr'],
                'prepositions': ['von', 'in', 'zu', 'für', 'mit', 'auf', 'bei'],
                'common_words': ['und', 'oder', 'aber', 'dass', 'ist', 'war', 'sind', 'waren']
            },
            'french': {
                'articles': ['le', 'la', 'les', 'un', 'une', 'des', 'du', 'de'],
                'pronouns': ['je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles'],
                'prepositions': ['de', 'dans', 'à', 'pour', 'avec', 'sur', 'par'],
                'common_words': ['et', 'ou', 'mais', 'que', 'ce', 'est', 'était', 'sont', 'étaient']
            },
            'dutch': {
                'articles': ['de', 'het', 'een'],
                'pronouns': ['ik', 'je', 'hij', 'zij', 'het', 'wij', 'jullie', 'zij'],
                'prepositions': ['van', 'in', 'naar', 'voor', 'met', 'op', 'bij'],
                'common_words': ['en', 'of', 'maar', 'dat', 'dit', 'is', 'was', 'zijn', 'waren']
            }
        }
        
        # Score each language based on indicator word presence
        language_scores = {}
        total_words = len(words)
        
        for lang, categories in language_indicators.items():
            score = 0
            for category, indicator_words in categories.items():
                matches = sum(1 for word in words if word in indicator_words)
                # Weight different categories
                if category == 'articles':
                    score += matches * 3
                elif category == 'pronouns':
                    score += matches * 2
                else:
                    score += matches
            
            # Normalize by text length
            language_scores[lang] = score / max(total_words, 1)
        
        # Check for non-ASCII characters (indicating potential non-English)
        has_accented_chars = any(ord(char) > 127 for char in text)
        
        # Determine most likely language
        if language_scores:
            best_lang = max(language_scores, key=language_scores.get)
            best_score = language_scores[best_lang]
            
            # Confidence threshold
            if best_score > 0.1:
                return best_lang
            elif has_accented_chars:
                return 'multilingual'
            else:
                return 'english'  # Default fallback
        
        return 'multilingual' if has_accented_chars else 'english'
    
    
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
        Complete multilingual analysis with both sentiment and aspects.
        Supports English, Spanish, German, French, and Dutch.
        
        The analysis pipeline:
        1. Detects language automatically
        2. Analyzes sentiment using language-specific features
        3. Classifies aspects with sentiment context
        4. Combines results with business intelligence
        """
        # Step 1: Sentiment analysis with automatic language detection
        sentiment_result = self.sentiment_classifier.analyze_sentiment(text)
        
        # Step 2: Aspect analysis with multilingual context
        aspect_result = self.aspect_classifier.classify_aspects_multilabel(
            text=text,
            language=sentiment_result.get('language', 'multilingual'),  # Default to multilingual instead of 'en'
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
    
    test_texts = [
        # English with mixed features
        "This product is absolutely fantastic and works perfectly!",
        
        # Spanish with negations and intensifiers
        "La aplicación no es nada útil, totalmente inútil y horrible.",
        
        # German with superlatives
        "Das ist das schlechteste Produkt aller Zeiten, völlig nutzlos.",
        
        # French with profanity and emphasis
        "C'est vraiment de la merde! Absolument terrible!!!",
        
        # Dutch with mixed sentiment
        "Het is niet verschrikkelijk, maar ook niet bijzonder geweldig."
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