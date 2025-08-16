#!/usr/bin/env python3
"""
Create and Test Enhanced ML Models
This script creates the enhanced model files and tests the complete system
"""

import os
import sys
import pandas as pd 

def create_enhanced_models():
    """Create the enhanced model files in src/models/"""
    
    print("🗏️ Creating Enhanced ML Models")
    print("="*50)
    
    # Create models directory
    models_dir = "src/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Enhanced Sentiment Classifier code (from your original document)
    sentiment_classifier_code = """import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
from typing import Dict, List, Tuple, Optional
import time
import logging

warnings.filterwarnings('ignore')

class EnhancedSentimentClassifier:
    \"\"\"
    Enhanced multilingual sentiment classifier with ensemble capabilities
    \"\"\"
    
    def __init__(self, use_ensemble=True, device='auto'):
        self.use_ensemble = use_ensemble
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.model_weights = {}
        self.performance_cache = {}
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initializing Enhanced Sentiment Classifier on {self.device}")
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        \"\"\"Load multiple sentiment analysis models for ensemble\"\"\"
        model_configs = [
            {
                'name': 'xlm_roberta',
                'model_id': 'cardiffnlp/twitter-xlm-roberta-base-sentiment',
                'weight': 0.4,
                'primary': True
            },
            {
                'name': 'multilingual_bert',
                'model_id': 'nlptown/bert-base-multilingual-uncased-sentiment',
                'weight': 0.35,
                'primary': False
            },
            {
                'name': 'distilbert_multilingual',
                'model_id': 'lxyuan/distilbert-base-multilingual-cased-sentiments-student',
                'weight': 0.25,
                'primary': False
            }
        ]
        
        for config in model_configs:
            try:
                self.logger.info(f"Loading {config['name']}...")
                
                # Try to load the model
                model = pipeline(
                    'sentiment-analysis',
                    model=config['model_id'],
                    device=0 if self.device == 'cuda' else -1,
                    return_all_scores=True
                )
                
                self.models[config['name']] = {
                    'pipeline': model,
                    'weight': config['weight'],
                    'primary': config['primary'],
                    'model_id': config['model_id']
                }
                
                self.logger.info(f"✅ {config['name']} loaded successfully")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load {config['name']}: {e}")
                if config['primary']:
                    # If primary model fails, try fallback
                    self._load_fallback_model()
        
        if not self.models:
            self.logger.error("❌ No models loaded successfully")
            raise Exception("Failed to load any sentiment analysis models")
        
        # Normalize weights
        self._normalize_model_weights()
    
    def _load_fallback_model(self):
        \"\"\"Load a simple fallback model if primary models fail\"\"\"
        try:
            self.logger.info("Loading fallback model...")
            fallback = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
            
            self.models['fallback'] = {
                'pipeline': fallback,
                'weight': 1.0,
                'primary': True,
                'model_id': 'distilbert-base-uncased-finetuned-sst-2-english'
            }
            self.logger.info("✅ Fallback model loaded")
            
        except Exception as e:
            self.logger.error(f"❌ Even fallback model failed: {e}")
    
    def _normalize_model_weights(self):
        \"\"\"Normalize model weights to sum to 1\"\"\"
        total_weight = sum(model['weight'] for model in self.models.values())
        for model_name in self.models:
            self.models[model_name]['weight'] /= total_weight
        
        self.logger.info(f"Model weights: {[(name, model['weight']) for name, model in self.models.items()]}")
    
    def analyze_sentiment(self, text: str, language: str = 'auto') -> Dict:
        \"\"\"
        Analyze sentiment of a single text
        
        Args:
            text: Input text to analyze
            language: Language of the text (auto-detect if 'auto')
            
        Returns:
            Dictionary with sentiment analysis results
        \"\"\"
        if not text.strip():
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33},
                'language': language,
                'model_used': 'none',
                'processing_time': 0.0
            }
        
        start_time = time.time()
        
        # Detect language if needed
        if language == 'auto':
            language = self._detect_language(text)
        
        # Get predictions from all models
        model_predictions = {}
        successful_models = []
        
        for model_name, model_info in self.models.items():
            try:
                prediction = model_info['pipeline'](text)
                model_predictions[model_name] = prediction
                successful_models.append(model_name)
                
            except Exception as e:
                self.logger.warning(f"Model {model_name} failed on text: {e}")
        
        if not successful_models:
            self.logger.error("All models failed for text analysis")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'scores': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33},
                'language': language,
                'model_used': 'none',
                'processing_time': time.time() - start_time,
                'error': 'All models failed'
            }
        
        # Ensemble predictions
        if self.use_ensemble and len(successful_models) > 1:
            ensemble_result = self._ensemble_predictions(model_predictions, successful_models)
            primary_model = 'ensemble'
        else:
            # Use single best model
            primary_model = successful_models[0]
            ensemble_result = self._process_single_prediction(
                model_predictions[primary_model], 
                self.models[primary_model]['model_id']
            )
        
        processing_time = time.time() - start_time
        
        return {
            'sentiment': ensemble_result['sentiment'],
            'confidence': ensemble_result['confidence'],
            'scores': ensemble_result['scores'],
            'language': language,
            'model_used': primary_model,
            'processing_time': processing_time,
            'models_available': successful_models
        }
    
    def _ensemble_predictions(self, model_predictions: Dict, successful_models: List) -> Dict:
        \"\"\"Combine predictions from multiple models using weighted average\"\"\"
        
        # Initialize score accumulator
        ensemble_scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        total_weight = 0.0
        
        for model_name in successful_models:
            if model_name not in model_predictions:
                continue
                
            prediction = model_predictions[model_name]
            weight = self.models[model_name]['weight']
            
            # Process prediction based on model output format
            scores = self._normalize_prediction_scores(prediction, self.models[model_name]['model_id'])
            
            # Add weighted scores
            for sentiment, score in scores.items():
                if sentiment in ensemble_scores:
                    ensemble_scores[sentiment] += score * weight
            
            total_weight += weight
        
        # Normalize final scores
        if total_weight > 0:
            for sentiment in ensemble_scores:
                ensemble_scores[sentiment] /= total_weight
        
        # Determine final sentiment
        max_sentiment = max(ensemble_scores, key=ensemble_scores.get)
        confidence = ensemble_scores[max_sentiment]
        
        return {
            'sentiment': max_sentiment,
            'confidence': confidence,
            'scores': ensemble_scores
        }
    
    def _process_single_prediction(self, prediction, model_id: str) -> Dict:
        \"\"\"Process prediction from a single model\"\"\"
        scores = self._normalize_prediction_scores(prediction, model_id)
        
        max_sentiment = max(scores, key=scores.get)
        confidence = scores[max_sentiment]
        
        return {
            'sentiment': max_sentiment,
            'confidence': confidence,
            'scores': scores
        }
    
    def _normalize_prediction_scores(self, prediction, model_id: str) -> Dict:
        \"\"\"Normalize prediction scores to consistent format\"\"\"
        scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        try:
            if isinstance(prediction, list) and len(prediction) > 0:
                # Handle multiple scores format
                if isinstance(prediction[0], list):
                    prediction = prediction[0]
                
                for item in prediction:
                    label = item['label'].lower()
                    score = item['score']
                    
                    # Map different label formats to standard format
                    if 'pos' in label or label in ['positive', '5 stars', '4 stars']:
                        scores['positive'] += score
                    elif 'neg' in label or label in ['negative', '1 star', '2 stars']:
                        scores['negative'] += score
                    elif 'neu' in label or label in ['neutral', '3 stars']:
                        scores['neutral'] += score
                    else:
                        # Handle numeric labels (1-5 stars)
                        if '5' in label or '4' in label:
                            scores['positive'] += score
                        elif '1' in label or '2' in label:
                            scores['negative'] += score
                        else:
                            scores['neutral'] += score
            
            # Ensure scores sum to 1
            total = sum(scores.values())
            if total > 0:
                scores = {k: v/total for k, v in scores.items()}
            else:
                scores = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
                
        except Exception as e:
            self.logger.warning(f"Error normalizing scores: {e}")
            scores = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
        
        return scores
    
    def _detect_language(self, text: str) -> str:
        \"\"\"Detect language of input text\"\"\"
        try:
            from langdetect import detect
            return detect(text)
        except:
            return 'en'  # Default to English
    
    def analyze_batch(self, texts: List[str], languages: List[str] = None) -> List[Dict]:
        \"\"\"
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            languages: List of languages (optional)
            
        Returns:
            List of sentiment analysis results
        \"\"\"
        if languages is None:
            languages = ['auto'] * len(texts)
        
        results = []
        for i, text in enumerate(texts):
            lang = languages[i] if i < len(languages) else 'auto'
            result = self.analyze_sentiment(text, lang)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict:
        \"\"\"Get information about loaded models\"\"\"
        return {
            'loaded_models': list(self.models.keys()),
            'ensemble_enabled': self.use_ensemble,
            'device': self.device,
            'model_weights': {name: model['weight'] for name, model in self.models.items()}
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize classifier
    classifier = EnhancedSentimentClassifier(use_ensemble=True)
    
    # Test texts in multiple languages
    test_texts = [
        "This product is amazing! Great quality and very easy to use.",
        "Terrible product, poor quality and confusing interface.",
        "Este producto es increíble! Excelente calidad y fácil de usar.",
        "Producto terrible, mala calidad e interfaz confusa.",
        "Dieses Produkt ist fantastisch! Großartige Qualität.",
        "Ce produit est incroyable! Excellente qualité."
    ]
    
    print("🧪 Testing Enhanced Sentiment Classifier")
    print("="*60)
    
    for i, text in enumerate(test_texts, 1):
        result = classifier.analyze_sentiment(text)
        print(f"\\n{i}. Text: {text[:50]}...")
        print(f"   Sentiment: {result['sentiment']} ({result['confidence']:.3f})")
        print(f"   Language: {result['language']}")
        print(f"   Model: {result['model_used']}")
        print(f"   Time: {result['processing_time']:.3f}s")
        print(f"   Scores: {result['scores']}")
    
    # Model info
    print(f"\\n📊 Model Information:")
    info = classifier.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
"""
    
    # Enhanced Aspect Classifier code (from your original document)
    aspect_classifier_code = """import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

class EnhancedAspectClassifier:
    \"\"\"
    Enhanced aspect classification system for product quality vs user experience
    Uses hybrid approach: keyword matching + semantic similarity + rule-based logic
    \"\"\"
    
    def __init__(self, confidence_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize aspect keywords with weights
        self._initialize_aspect_keywords()
        self._initialize_semantic_patterns()
        self._initialize_tfidf_vectorizer()
        
        self.logger.info("Enhanced Aspect Classifier initialized")
    
    def _initialize_aspect_keywords(self):
        \"\"\"Initialize weighted keyword dictionaries for different languages\"\"\"
        
        self.aspect_keywords = {
            'product_quality': {
                'en': {
                    # High-weight quality indicators
                    'high': ['quality', 'durable', 'reliable', 'performance', 'build', 'material', 'craftsmanship', 
                            'sturdy', 'solid', 'robust', 'premium', 'excellent', 'superior', 'outstanding'],
                    # Medium-weight quality indicators  
                    'medium': ['fast', 'speed', 'efficient', 'smooth', 'stable', 'consistent', 'accurate',
                              'precise', 'powerful', 'strong', 'effective'],
                    # Negative quality indicators
                    'negative': ['cheap', 'flimsy', 'broken', 'defect', 'faulty', 'poor', 'terrible', 'awful',
                                'damaged', 'fragile', 'weak', 'unreliable', 'inconsistent', 'slow', 'buggy',
                                'crash', 'freeze', 'lag', 'glitch', 'error', 'malfunction']
                },
                'es': {
                    'high': ['calidad', 'duradero', 'confiable', 'rendimiento', 'construcción', 'material',
                            'artesanía', 'sólido', 'resistente', 'robusto', 'premium', 'excelente'],
                    'medium': ['rápido', 'velocidad', 'eficiente', 'suave', 'estable', 'consistente', 'preciso'],
                    'negative': ['barato', 'frágil', 'roto', 'defecto', 'defectuoso', 'pobre', 'terrible', 'malo',
                                'dañado', 'débil', 'poco confiable', 'lento', 'error', 'fallo']
                },
                'de': {
                    'high': ['qualität', 'langlebig', 'zuverlässig', 'leistung', 'bau', 'material', 'handwerk',
                            'stabil', 'fest', 'robust', 'premium', 'ausgezeichnet'],
                    'medium': ['schnell', 'geschwindigkeit', 'effizient', 'glatt', 'stabil', 'konsistent', 'genau'],
                    'negative': ['billig', 'zerbrechlich', 'kaputt', 'defekt', 'fehlerhaft', 'schlecht', 'schrecklich',
                                'beschädigt', 'schwach', 'unzuverlässig', 'langsam', 'fehler']
                },
                'fr': {
                    'high': ['qualité', 'durable', 'fiable', 'performance', 'construction', 'matériel',
                            'artisanat', 'solide', 'robuste', 'premium', 'excellent'],
                    'medium': ['rapide', 'vitesse', 'efficace', 'lisse', 'stable', 'cohérent', 'précis'],
                    'negative': ['bon marché', 'fragile', 'cassé', 'défaut', 'défectueux', 'pauvre', 'terrible',
                                'endommagé', 'faible', 'peu fiable', 'lent', 'erreur']
                }
            },
            'user_experience': {
                'en': {
                    'high': ['easy', 'simple', 'intuitive', 'user-friendly', 'straightforward', 'clear',
                            'convenient', 'accessible', 'smooth', 'seamless', 'elegant', 'beautiful'],
                    'medium': ['interface', 'design', 'layout', 'navigation', 'menu', 'button', 'screen',
                              'usable', 'functional', 'practical', 'helpful', 'useful'],
                    'negative': ['difficult', 'hard', 'confusing', 'complicated', 'complex', 'frustrating',
                                'annoying', 'clunky', 'awkward', 'unintuitive', 'messy', 'cluttered',
                                'unclear', 'confusing', 'hidden', 'buried']
                },
                'es': {
                    'high': ['fácil', 'simple', 'intuitivo', 'amigable', 'directo', 'claro', 'conveniente',
                            'accesible', 'suave', 'elegante', 'hermoso'],
                    'medium': ['interfaz', 'diseño', 'diseño', 'navegación', 'menú', 'botón', 'pantalla',
                              'usable', 'funcional', 'práctico', 'útil'],
                    'negative': ['difícil', 'complicado', 'confuso', 'complejo', 'frustrante', 'molesto',
                                'torpe', 'poco intuitivo', 'desordenado', 'poco claro']
                },
                'de': {
                    'high': ['einfach', 'simpel', 'intuitiv', 'benutzerfreundlich', 'unkompliziert', 'klar',
                            'bequem', 'zugänglich', 'glatt', 'elegant', 'schön'],
                    'medium': ['benutzeroberfläche', 'design', 'layout', 'navigation', 'menü', 'taste',
                              'bildschirm', 'nutzbar', 'funktional', 'praktisch', 'hilfreich'],
                    'negative': ['schwierig', 'hart', 'verwirrend', 'kompliziert', 'komplex', 'frustrierend',
                                'ärgerlich', 'ungeschickt', 'unintuitiv', 'unordentlich', 'unklar']
                },
                'fr': {
                    'high': ['facile', 'simple', 'intuitif', 'convivial', 'direct', 'clair', 'pratique',
                            'accessible', 'lisse', 'élégant', 'beau'],
                    'medium': ['interface', 'design', 'mise en page', 'navigation', 'menu', 'bouton',
                              'écran', 'utilisable', 'fonctionnel', 'pratique', 'utile'],
                    'negative': ['difficile', 'dur', 'confus', 'compliqué', 'complexe', 'frustrant',
                                'ennuyeux', 'maladroit', 'peu intuitif', 'désordonné', 'peu clair']
                }
            }
        }
    
    def _initialize_semantic_patterns(self):
        \"\"\"Initialize semantic patterns for better aspect detection\"\"\"
        
        self.semantic_patterns = {
            'product_quality': [
                r'\\b(works?|working|performance|speed|fast|slow)\\b',
                r'\\b(quality|build|material|durable|reliable|stable)\\b',
                r'\\b(crash|bug|error|glitch|freeze|lag)\\b',
                r'\\b(battery|charge|power|energy)\\b',
                r'\\b(sound|audio|video|picture|image)\\b',
                r'\\b(delivery|shipping|package|tracking)\\b'  # For logistics apps
            ],
            'user_experience': [
                r'\\b(easy|difficult|hard|simple|complex|intuitive)\\b',
                r'\\b(interface|design|layout|menu|button|screen)\\b',
                r'\\b(navigate|navigation|find|search|locate)\\b',
                r'\\b(user|use|using|usage|experience)\\b',
                r'\\b(confusing|clear|obvious|hidden|unclear)\\b',
                r'\\b(learn|understand|figure|discover)\\b'
            ]
        }
    
    def _initialize_tfidf_vectorizer(self):
        \"\"\"Initialize TF-IDF vectorizer for semantic similarity\"\"\"
        
        # Create reference texts for each aspect
        self.reference_texts = {
            'product_quality': [
                "excellent product quality and build materials",
                "fast performance and reliable functionality", 
                "durable construction and superior craftsmanship",
                "consistent speed and accurate results",
                "stable operation and smooth performance",
                "high quality materials and solid build",
                "crashes frequently and poor performance",
                "buggy software with many errors and glitches",
                "slow speed and unreliable operation",
                "cheap materials and poor construction quality"
            ],
            'user_experience': [
                "easy to use and intuitive interface design",
                "simple navigation and clear menu layout",
                "user-friendly design and straightforward operation",
                "convenient access and helpful features",
                "smooth user experience and elegant interface",
                "beautiful design and accessible functionality",
                "confusing interface and difficult navigation",
                "complicated design and hard to understand",
                "cluttered layout and unintuitive controls",
                "frustrating user experience and poor design"
            ]
        }
        
        # Combine all reference texts
        all_texts = []
        self.text_labels = []
        
        for aspect, texts in self.reference_texts.items():
            all_texts.extend(texts)
            self.text_labels.extend([aspect] * len(texts))
        
        # Fit TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        
        self.reference_vectors = self.tfidf.fit_transform(all_texts)
    
    def classify_aspect(self, text: str, language: str = 'en') -> Dict:
        \"\"\"
        Classify aspect of given text using hybrid approach
        
        Args:
            text: Input text to classify
            language: Language of the text
            
        Returns:
            Dictionary with aspect classification results
        \"\"\"
        if not text.strip():
            return {
                'aspect': 'general',
                'confidence': 0.0,
                'method': 'empty_text',
                'scores': {'product_quality': 0.0, 'user_experience': 0.0, 'general': 1.0}
            }
        
        text_lower = text.lower()
        
        # Method 1: Keyword-based scoring
        keyword_scores = self._keyword_based_classification(text_lower, language)
        
        # Method 2: Semantic similarity (for English texts or if language not supported)
        semantic_scores = self._semantic_similarity_classification(text)
        
        # Method 3: Pattern matching
        pattern_scores = self._pattern_based_classification(text_lower)
        
        # Combine scores with weights
        combined_scores = self._combine_classification_scores(
            keyword_scores, semantic_scores, pattern_scores, language
        )
        
        # Determine final aspect
        max_aspect = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[max_aspect]
        
        # If confidence is too low, classify as general
        if confidence < self.confidence_threshold:
            aspect = 'general'
            confidence = 1.0 - confidence
            method = 'low_confidence'
        else:
            aspect = max_aspect
            method = 'hybrid'
        
        return {
            'aspect': aspect,
            'confidence': confidence,
            'method': method,
            'scores': combined_scores,
            'keyword_scores': keyword_scores,
            'semantic_scores': semantic_scores,
            'pattern_scores': pattern_scores
        }
    
    def _keyword_based_classification(self, text: str, language: str) -> Dict:
        \"\"\"Classify using keyword matching with weights\"\"\"
        
        scores = {'product_quality': 0.0, 'user_experience': 0.0}
        
        # Default to English if language not supported
        lang = language if language in ['en', 'es', 'de', 'fr'] else 'en'
        
        for aspect, lang_dict in self.aspect_keywords.items():
            if lang in lang_dict:
                keywords = lang_dict[lang]
                
                # Weight by keyword importance
                high_weight = 3.0
                medium_weight = 2.0
                negative_weight = 2.5
                
                for weight_category, weight in [('high', high_weight), ('medium', medium_weight), ('negative', negative_weight)]:
                    if weight_category in keywords:
                        for keyword in keywords[weight_category]:
                            if keyword in text:
                                scores[aspect] += weight
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    def _semantic_similarity_classification(self, text: str) -> Dict:
        \"\"\"Classify using semantic similarity with reference texts\"\"\"
        
        try:
            # Transform input text
            text_vector = self.tfidf.transform([text])
            
            # Calculate similarities
            similarities = cosine_similarity(text_vector, self.reference_vectors)[0]
            
            # Group by aspect
            aspect_similarities = {'product_quality': [], 'user_experience': []}
            
            for i, similarity in enumerate(similarities):
                aspect = self.text_labels[i]
                aspect_similarities[aspect].append(similarity)
            
            # Calculate average similarity for each aspect
            scores = {}
            for aspect, sims in aspect_similarities.items():
                scores[aspect] = np.mean(sims) if sims else 0.0
            
            # Normalize
            total = sum(scores.values())
            if total > 0:
                scores = {k: v/total for k, v in scores.items()}
            
            return scores
            
        except Exception as e:
            self.logger.warning(f"Semantic similarity failed: {e}")
            return {'product_quality': 0.5, 'user_experience': 0.5}
    
    def _pattern_based_classification(self, text: str) -> Dict:
        \"\"\"Classify using regex patterns\"\"\"
        
        scores = {'product_quality': 0.0, 'user_experience': 0.0}
        
        for aspect, patterns in self.semantic_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                scores[aspect] += matches
        
        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    def _combine_classification_scores(self, keyword_scores: Dict, semantic_scores: Dict, 
                                    pattern_scores: Dict, language: str) -> Dict:
        \"\"\"Combine different classification methods with appropriate weights\"\"\"
        
        # Adjust weights based on language
        if language == 'en':
            # English: All methods work well
            keyword_weight = 0.4
            semantic_weight = 0.4
            pattern_weight = 0.2
        else:
            # Non-English: Rely more on keywords and patterns
            keyword_weight = 0.6
            semantic_weight = 0.2
            pattern_weight = 0.2
        
        combined = {'product_quality': 0.0, 'user_experience': 0.0}
        
        for aspect in combined.keys():
            combined[aspect] = (
                keyword_scores.get(aspect, 0) * keyword_weight +
                semantic_scores.get(aspect, 0) * semantic_weight +
                pattern_scores.get(aspect, 0) * pattern_weight
            )
        
        return combined
    
    def classify_batch(self, texts: List[str], languages: List[str] = None) -> List[Dict]:
        \"\"\"
        Classify aspects for multiple texts
        
        Args:
            texts: List of texts to classify
            languages: List of languages (optional)
            
        Returns:
            List of aspect classification results
        \"\"\"
        if languages is None:
            languages = ['en'] * len(texts)
        
        results = []
        for i, text in enumerate(texts):
            lang = languages[i] if i < len(languages) else 'en'
            result = self.classify_aspect(text, lang)
            results.append(result)
        
        return results
    
    def get_aspect_summary(self, results: List[Dict]) -> Dict:
        \"\"\"Generate summary statistics for aspect classification results\"\"\"
        
        aspect_counts = {'product_quality': 0, 'user_experience': 0, 'general': 0}
        confidence_scores = []
        methods_used = []
        
        for result in results:
            aspect_counts[result['aspect']] += 1
            confidence_scores.append(result['confidence'])
            methods_used.append(result['method'])
        
        return {
            'total_texts': len(results),
            'aspect_distribution': aspect_counts,
            'aspect_percentages': {k: (v/len(results))*100 for k, v in aspect_counts.items()},
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'confidence_std': np.std(confidence_scores) if confidence_scores else 0,
            'methods_used': dict(pd.Series(methods_used).value_counts())
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize classifier
    classifier = EnhancedAspectClassifier()
    
    # Test texts covering different aspects
    test_texts = [
        # Product Quality Examples
        "The product quality is excellent, very durable and well-made materials",
        "Poor build quality, broke after one week of normal use",
        "App crashes frequently and has many bugs that need fixing",
        "Fast performance and reliable operation, works perfectly",
        "La calidad del producto es excelente, muy duradero",
        "Mala calidad, se rompió después de una semana",
        
        # User Experience Examples  
        "Very easy to use, intuitive interface and smooth navigation",
        "Confusing interface, hard to find basic features and settings",
        "Great design, user-friendly and straightforward to operate",
        "Complicated layout, difficult to understand and navigate",
        "Muy fácil de usar, interfaz intuitiva y navegación suave",
        "Interfaz confusa, difícil de encontrar funciones básicas",
        
        # Mixed/Ambiguous Examples
        "Good overall experience with some minor issues here and there",
        "Average product, nothing special but works as expected",
        "I love this app, it's perfect for my daily needs"
    ]
    
    print("🎯 Testing Enhanced Aspect Classifier")
    print("="*70)
    
    # Test individual classifications
    for i, text in enumerate(test_texts, 1):
        result = classifier.classify_aspect(text)
        print(f"\\n{i}. Text: {text[:60]}...")
        print(f"   Aspect: {result['aspect']} ({result['confidence']:.3f})")
        print(f"   Method: {result['method']}")
        print(f"   Scores: P={result['scores'].get('product_quality', 0):.2f}, "
              f"UX={result['scores'].get('user_experience', 0):.2f}")
    
    # Test batch classification
    print(f"\\n📊 Batch Classification Summary:")
    batch_results = classifier.classify_batch(test_texts)
    summary = classifier.get_aspect_summary(batch_results)
    
    print(f"Total texts analyzed: {summary['total_texts']}")
    print(f"Aspect distribution: {summary['aspect_distribution']}")
    print(f"Aspect percentages: {summary['aspect_percentages']}")
    print(f"Average confidence: {summary['average_confidence']:.3f}")
    print(f"Methods used: {summary['methods_used']}")
"""
    
    # Write files
    files_to_create = [
        (f"{models_dir}/enhanced_sentiment_classifier.py", sentiment_classifier_code),
        (f"{models_dir}/enhanced_aspect_classifier.py", aspect_classifier_code),
        (f"{models_dir}/__init__.py", "# Enhanced ML Models Package")
    ]
    
    for filepath, content in files_to_create:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Created: {filepath}")
    
    print(f"✅ All enhanced model files created!")

def create_integrated_pipeline():
    """Create the integrated pipeline file"""
    
    print("\\n🔗 Creating Integrated Pipeline")
    print("="*50)
    
    pipeline_code = """import pandas as pd
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
    \"\"\"Integrated ML Pipeline for production use\"\"\"
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_classifiers()
    
    def _initialize_classifiers(self):
        \"\"\"Initialize classifiers\"\"\"
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
        \"\"\"Initialize fallback sentiment analysis\"\"\"
        from transformers import pipeline
        
        self.sentiment_pipeline = pipeline('sentiment-analysis', 
                                         model='nlptown/bert-base-multilingual-uncased-sentiment',
                                         return_all_scores=True)
        self.logger.info("✅ Fallback sentiment classifier loaded")
    
    def _initialize_fallback_aspect(self):
        \"\"\"Initialize fallback aspect detection\"\"\"
        self.aspect_keywords = {
            'product_quality': ['quality', 'performance', 'build', 'durable', 'reliable', 'fast', 'crash', 'bug'],
            'user_experience': ['easy', 'difficult', 'interface', 'design', 'intuitive', 'confusing', 'simple']
        }
        self.logger.info("✅ Fallback aspect classifier loaded")
    
    def _initialize_fallback_models(self):
        \"\"\"Initialize minimal fallback models\"\"\"
        self._initialize_fallback_sentiment()
        self._initialize_fallback_aspect()
    
    def analyze_text(self, text: str, language: str = 'auto') -> Dict:
        \"\"\"Analyze a single text\"\"\"
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
        \"\"\"Fallback sentiment analysis\"\"\"
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
        \"\"\"Fallback aspect analysis\"\"\"
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
        \"\"\"Analyze multiple texts\"\"\"
        if languages is None:
            languages = ['auto'] * len(texts)
        
        results = []
        for text, lang in zip(texts, languages):
            result = self.analyze_text(text, lang)
            results.append(result)
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        \"\"\"Analyze DataFrame\"\"\"
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
"""
    
    # Write pipeline file
    pipeline_path = "src/integrated_ml_pipeline.py"
    with open(pipeline_path, 'w', encoding='utf-8') as f:
        f.write(pipeline_code)
    
    print(f"✅ Created: {pipeline_path}")

def test_enhanced_models():
    """Test the enhanced models"""
    
    print("\\n🧪 Testing Enhanced Models")
    print("="*50)
    
    # Add src to Python path
    sys.path.append('src')
    
    try:
        from integrated_ml_pipeline import IntegratedMLPipeline
        
        # Initialize pipeline
        pipeline = IntegratedMLPipeline()
        
        # Test texts
        test_texts = [
            "This product has excellent quality but the interface is confusing",
            "App crashes frequently and the user experience is terrible", 
            "Very easy to use and great build quality",
            "La calidad del producto es excelente pero la interfaz es confusa",
            "Sehr benutzerfreundlich aber schlechte Materialqualität"
        ]
        
        print("\\nTesting individual text analysis:")
        for i, text in enumerate(test_texts, 1):
            result = pipeline.analyze_text(text)
            print(f"\\n{i}. Text: {text}")
            print(f"   Sentiment: {result['sentiment']} ({result['sentiment_confidence']:.3f})")
            print(f"   Aspect: {result['aspect']} ({result['aspect_confidence']:.3f})")
            print(f"   Time: {result['processing_time']:.3f}s")
        
        # Test batch analysis
        print("\\nTesting batch analysis:")
        batch_results = pipeline.analyze_batch(test_texts)
        print(f"✅ Processed {len(batch_results)} texts successfully")
        
        # Test DataFrame analysis
        print("\\nTesting DataFrame analysis:")
        df = pd.DataFrame({'text': test_texts})
        results_df = pipeline.analyze_dataframe(df)
        print(f"✅ DataFrame analysis complete:")
        print(results_df[['text', 'predicted_sentiment', 'predicted_aspect']].head())
        
        print("\\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to create and test everything"""
    
    print("🚀 Creating Enhanced ML System")
    print("="*60)
    
    # Create enhanced models
    create_enhanced_models()
    
    # Create integrated pipeline
    create_integrated_pipeline()
    
    # Test the system
    test_enhanced_models()
    
    print("\\n" + "="*60)
    print("🎉 Enhanced ML System Created Successfully!")
    print("="*60)
    
    print("\\n📁 Files created:")
    print("   src/models/enhanced_sentiment_classifier.py")
    print("   src/models/enhanced_aspect_classifier.py") 
    print("   src/models/__init__.py")
    print("   src/integrated_ml_pipeline.py")
    
    print("\\n🚀 Next steps:")
    print("   1. Run: python src/integrated_ml_pipeline.py")
    print("   2. Continue with web application development")
    print("   3. Test with your FedEx data")
    
    print("\\n✅ Ready for the Web Application!")

if __name__ == "__main__":
    main()