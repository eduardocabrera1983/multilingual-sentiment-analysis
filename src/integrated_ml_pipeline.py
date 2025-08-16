import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import time
import logging
import warnings
import json
import os
from datetime import datetime

# Import our custom classifiers
# from enhanced_sentiment_classifier import EnhancedSentimentClassifier
# from enhanced_aspect_classifier import EnhancedAspectClassifier

warnings.filterwarnings('ignore')

class IntegratedMLPipeline:
    """
    Integrated ML Pipeline for Multilingual Sentiment Analysis with Aspect Detection
    Combines sentiment analysis and aspect classification for production use
    """
    
    def __init__(self, 
                 use_ensemble=True, 
                 confidence_threshold=0.3,
                 cache_results=True,
                 save_results=True):
        
        self.use_ensemble = use_ensemble
        self.confidence_threshold = confidence_threshold
        self.cache_results = cache_results
        self.save_results = save_results
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize classifiers
        self._initialize_classifiers()
        
        # Initialize result storage
        self.results_cache = {}
        self.batch_results = []
        
        self.logger.info("Integrated ML Pipeline initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_classifiers(self):
        """Initialize sentiment and aspect classifiers"""
        try:
            self.logger.info("Loading sentiment classifier...")
            # For now, we'll use a simplified version that works with transformers
            from transformers import pipeline
            
            # Load sentiment analysis pipeline
            try:
                self.sentiment_pipeline = pipeline(
                    'sentiment-analysis',
                    model='nlptown/bert-base-multilingual-uncased-sentiment',
                    return_all_scores=True
                )
                self.logger.info("✅ Sentiment classifier loaded")
            except Exception as e:
                self.logger.warning(f"Primary sentiment model failed: {e}")
                self.sentiment_pipeline = pipeline('sentiment-analysis')
                self.logger.info("✅ Fallback sentiment classifier loaded")
            
            # Initialize aspect classifier (simplified version)
            self._initialize_aspect_detection()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize classifiers: {e}")
            raise
    
    def _initialize_aspect_detection(self):
        """Initialize aspect detection with keyword-based approach"""
        
        self.aspect_keywords = {
            'product_quality': {
                'en': ['quality', 'durable', 'reliable', 'performance', 'build', 'material', 'craftsmanship',
                      'fast', 'speed', 'stable', 'crash', 'bug', 'error', 'broken', 'defect', 'poor'],
                'es': ['calidad', 'duradero', 'confiable', 'rendimiento', 'construcción', 'material',
                      'rápido', 'velocidad', 'estable', 'error', 'roto', 'defecto', 'pobre'],
                'de': ['qualität', 'langlebig', 'zuverlässig', 'leistung', 'bau', 'material',
                      'schnell', 'geschwindigkeit', 'stabil', 'fehler', 'kaputt', 'defekt'],
                'fr': ['qualité', 'durable', 'fiable', 'performance', 'construction', 'matériel',
                      'rapide', 'vitesse', 'stable', 'erreur', 'cassé', 'défaut']
            },
            'user_experience': {
                'en': ['easy', 'difficult', 'interface', 'design', 'navigate', 'intuitive', 'confusing',
                      'simple', 'complex', 'user', 'menu', 'button', 'screen', 'layout'],
                'es': ['fácil', 'difícil', 'interfaz', 'diseño', 'navegar', 'intuitivo', 'confuso',
                      'simple', 'complejo', 'usuario', 'menú', 'botón', 'pantalla'],
                'de': ['einfach', 'schwierig', 'benutzeroberfläche', 'design', 'navigieren', 'intuitiv',
                      'verwirrend', 'einfach', 'komplex', 'benutzer', 'menü', 'taste'],
                'fr': ['facile', 'difficile', 'interface', 'design', 'naviguer', 'intuitif', 'confus',
                      'simple', 'complexe', 'utilisateur', 'menu', 'bouton']
            }
        }
        
        self.logger.info("✅ Aspect detection initialized")
    
    def analyze_single_text(self, 
                           text: str, 
                           language: str = 'auto',
                           include_raw_scores: bool = False) -> Dict:
        """
        Analyze a single text for sentiment and aspect
        
        Args:
            text: Input text to analyze
            language: Language of the text ('auto' for auto-detection)
            include_raw_scores: Whether to include raw model scores
            
        Returns:
            Dictionary with complete analysis results
        """
        
        if not text or not text.strip():
            return self._create_empty_result()
        
        start_time = time.time()
        
        # Detect language if needed
        if language == 'auto':
            language = self._detect_language(text)
        
        # Check cache if enabled
        cache_key = f"{hash(text)}_{language}"
        if self.cache_results and cache_key in self.results_cache:
            self.logger.debug(f"Cache hit for text: {text[:50]}...")
            return self.results_cache[cache_key]
        
        # Analyze sentiment
        sentiment_result = self._analyze_sentiment(text, language)
        
        # Analyze aspect
        aspect_result = self._analyze_aspect(text, language)
        
        # Create unified result
        result = {
            'text': text,
            'language': language,
            'timestamp': datetime.now().isoformat(),
            
            # Sentiment results
            'sentiment': sentiment_result['sentiment'],
            'sentiment_confidence': sentiment_result['confidence'],
            'sentiment_scores': sentiment_result['scores'],
            
            # Aspect results
            'aspect': aspect_result['aspect'],
            'aspect_confidence': aspect_result['confidence'],
            'aspect_scores': aspect_result['scores'],
            
            # Meta information
            'processing_time': time.time() - start_time,
            'model_version': '1.0',
            'pipeline_version': self._get_pipeline_version()
        }
        
        # Add raw scores if requested
        if include_raw_scores:
            result['raw_sentiment_scores'] = sentiment_result.get('raw_scores', {})
            result['raw_aspect_scores'] = aspect_result.get('raw_scores', {})
        
        # Cache result
        if self.cache_results:
            self.results_cache[cache_key] = result
        
        return result
    
    def analyze_batch(self, 
                     texts: List[str], 
                     languages: List[str] = None,
                     show_progress: bool = True) -> List[Dict]:
        """
        Analyze multiple texts in batch
        
        Args:
            texts: List of texts to analyze
            languages: List of languages (optional)
            show_progress: Whether to show progress information
            
        Returns:
            List of analysis results
        """
        
        if languages is None:
            languages = ['auto'] * len(texts)
        elif len(languages) == 1:
            languages = languages * len(texts)
        
        results = []
        total_texts = len(texts)
        
        self.logger.info(f"Starting batch analysis of {total_texts} texts")
        
        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{total_texts} texts")
            
            language = languages[i] if i < len(languages) else 'auto'
            result = self.analyze_single_text(text, language)
            results.append(result)
        
        # Store batch results
        if self.save_results:
            self.batch_results.extend(results)
        
        self.logger.info(f"Batch analysis complete: {total_texts} texts processed")
        
        return results
    
    def analyze_dataframe(self, 
                         df: pd.DataFrame, 
                         text_column: str = 'text',
                         language_column: str = None) -> pd.DataFrame:
        """
        Analyze texts in a pandas DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column
            language_column: Name of the language column (optional)
            
        Returns:
            DataFrame with analysis results added
        """
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        texts = df[text_column].astype(str).tolist()
        
        if language_column and language_column in df.columns:
            languages = df[language_column].astype(str).tolist()
        else:
            languages = None
        
        # Analyze batch
        results = self.analyze_batch(texts, languages)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Merge with original DataFrame
        output_df = df.copy()
        
        # Add analysis columns
        analysis_columns = [
            'sentiment', 'sentiment_confidence', 'aspect', 'aspect_confidence',
            'processing_time', 'language'
        ]
        
        for col in analysis_columns:
            if col in results_df.columns:
                output_df[f'predicted_{col}'] = results_df[col]
        
        return output_df
    
    def _analyze_sentiment(self, text: str, language: str) -> Dict:
        """Analyze sentiment using the loaded pipeline"""
        
        try:
            # Get prediction from pipeline
            prediction = self.sentiment_pipeline(text)
            
            # Process results based on pipeline output format
            if isinstance(prediction, list) and len(prediction) > 0:
                if isinstance(prediction[0], list):
                    # Multiple scores format
                    scores = self._process_sentiment_scores(prediction[0])
                else:
                    # Single prediction format
                    scores = self._process_sentiment_scores(prediction)
            else:
                scores = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            
            # Determine sentiment
            max_sentiment = max(scores, key=scores.get)
            confidence = scores[max_sentiment]
            
            return {
                'sentiment': max_sentiment,
                'confidence': confidence,
                'scores': scores,
                'raw_scores': prediction
            }
            
        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'error': str(e)
            }
    
    def _process_sentiment_scores(self, prediction: List[Dict]) -> Dict:
        """Process sentiment scores from pipeline output"""
        
        scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        for item in prediction:
            label = item['label'].lower()
            score = item['score']
            
            # Map labels to standard format
            if any(x in label for x in ['pos', 'positive', '4', '5']):
                scores['positive'] += score
            elif any(x in label for x in ['neg', 'negative', '1', '2']):
                scores['negative'] += score
            else:
                scores['neutral'] += score
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores
    
    def _analyze_aspect(self, text: str, language: str) -> Dict:
        """Analyze aspect using keyword-based approach"""
        
        text_lower = text.lower()
        
        # Count keywords for each aspect
        scores = {'product_quality': 0, 'user_experience': 0}
        
        # Use English keywords if language not supported
        lang = language if language in self.aspect_keywords['product_quality'] else 'en'
        
        for aspect, lang_keywords in self.aspect_keywords.items():
            if lang in lang_keywords:
                for keyword in lang_keywords[lang]:
                    if keyword in text_lower:
                        scores[aspect] += 1
        
        # Determine aspect
        total_score = sum(scores.values())
        
        if total_score == 0:
            # No specific keywords found
            aspect = 'general'
            confidence = 0.5
            normalized_scores = {'product_quality': 0.0, 'user_experience': 0.0, 'general': 1.0}
        else:
            # Normalize scores
            normalized_scores = {k: v/total_score for k, v in scores.items()}
            normalized_scores['general'] = 0.0
            
            max_aspect = max(scores, key=scores.get)
            
            if normalized_scores[max_aspect] < self.confidence_threshold:
                aspect = 'general'
                confidence = 0.5
            else:
                aspect = max_aspect
                confidence = normalized_scores[max_aspect]
        
        return {
            'aspect': aspect,
            'confidence': confidence,
            'scores': normalized_scores,
            'raw_scores': scores
        }
    
    def _detect_language(self, text: str) -> str:
        """Detect language of input text"""
        try:
            from langdetect import detect
            return detect(text)
        except:
            return 'en'  # Default to English
    
    def _create_empty_result(self) -> Dict:
        """Create empty result for invalid input"""
        return {
            'text': '',
            'language': 'unknown',
            'timestamp': datetime.now().isoformat(),
            'sentiment': 'neutral',
            'sentiment_confidence': 0.0,
            'sentiment_scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
            'aspect': 'general',
            'aspect_confidence': 0.0,
            'aspect_scores': {'product_quality': 0.0, 'user_experience': 0.0, 'general': 1.0},
            'processing_time': 0.0,
            'model_version': '1.0',
            'error': 'Empty or invalid input'
        }
    
    def _get_pipeline_version(self) -> str:
        """Get pipeline version information"""
        return f"IntegratedML-v1.0-{datetime.now().strftime('%Y%m%d')}"
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics from processed results"""
        
        if not self.batch_results:
            return {'error': 'No batch results available'}
        
        df = pd.DataFrame(self.batch_results)
        
        stats = {
            'total_processed': len(df),
            'average_processing_time': df['processing_time'].mean(),
            'language_distribution': df['language'].value_counts().to_dict(),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'aspect_distribution': df['aspect'].value_counts().to_dict(),
            'average_sentiment_confidence': df['sentiment_confidence'].mean(),
            'average_aspect_confidence': df['aspect_confidence'].mean(),
            'cache_hit_rate': len(self.results_cache) / len(df) if len(df) > 0 else 0
        }
        
        return stats
    
    def save_results(self, filepath: str = None) -> str:
        """Save batch results to file"""
        
        if not self.batch_results:
            self.logger.warning("No results to save")
            return None
        
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"results/ml_pipeline_results_{timestamp}.json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(self.batch_results, f, indent=2)
        
        self.logger.info(f"Results saved to: {filepath}")
        return filepath
    
    def export_to_csv(self, filepath: str = None) -> str:
        """Export results to CSV format"""
        
        if not self.batch_results:
            self.logger.warning("No results to export")
            return None
        
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"results/ml_pipeline_results_{timestamp}.csv"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(self.batch_results)
        
        # Flatten nested dictionaries
        sentiment_scores_df = pd.json_normalize(df['sentiment_scores'])
        sentiment_scores_df.columns = [f'sentiment_score_{col}' for col in sentiment_scores_df.columns]
        
        aspect_scores_df = pd.json_normalize(df['aspect_scores'])
        aspect_scores_df.columns = [f'aspect_score_{col}' for col in aspect_scores_df.columns]
        
        # Combine DataFrames
        export_df = pd.concat([
            df.drop(['sentiment_scores', 'aspect_scores'], axis=1),
            sentiment_scores_df,
            aspect_scores_df
        ], axis=1)
        
        export_df.to_csv(filepath, index=False)
        
        self.logger.info(f"Results exported to CSV: {filepath}")
        return filepath

# Example usage and testing
if __name__ == "__main__":
    # Initialize pipeline
    print("🚀 Initializing Integrated ML Pipeline...")
    pipeline = IntegratedMLPipeline(use_ensemble=True, confidence_threshold=0.3)
    
    # Test single text analysis
    print("\n🧪 Testing Single Text Analysis:")
    print("="*60)
    
    test_texts = [
        "This product has excellent quality but the interface is confusing",
        "App crashes frequently and the user experience is terrible",
        "Very easy to use and great build quality",
        "La calidad del producto es excelente pero la interfaz es confusa",
        "Sehr benutzerfreundlich aber schlechte Materialqualität"
    ]
    
    for i, text in enumerate(test_texts, 1):
        result = pipeline.analyze_single_text(text)
        print(f"\n{i}. Text: {text}")
        print(f"   Language: {result['language']}")
        print(f"   Sentiment: {result['sentiment']} ({result['sentiment_confidence']:.3f})")
        print(f"   Aspect: {result['aspect']} ({result['aspect_confidence']:.3f})")
        print(f"   Time: {result['processing_time']:.3f}s")
    
    # Test batch analysis
    print(f"\n📊 Testing Batch Analysis:")
    print("="*60)
    
    batch_results = pipeline.analyze_batch(test_texts, show_progress=True)
    print(f"Processed {len(batch_results)} texts")
    
    # Get performance stats
    print(f"\n📈 Performance Statistics:")
    stats = pipeline.get_performance_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Save results
    print(f"\n💾 Saving Results:")
    json_file = pipeline.save_results()
    csv_file = pipeline.export_to_csv()
    print(f"   JSON: {json_file}")
    print(f"   CSV: {csv_file}")
    
    print("\n✅ Integrated ML Pipeline testing complete!")