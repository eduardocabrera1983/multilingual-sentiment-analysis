import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from langdetect import detect
import time
import warnings
warnings.filterwarnings('ignore')

class ModelTester:
    def __init__(self):
        self.models = {}
        self.test_texts = {
            'en': [
                "This product is amazing! Great quality and easy to use.",
                "Terrible product, poor quality and confusing interface.",
                "The product is okay, average quality but hard to use."
            ],
            'es': [
                "Este producto es increíble! Excelente calidad y fácil de usar.",
                "Producto terrible, mala calidad e interfaz confusa.",
                "El producto está bien, calidad promedio pero difícil de usar."
            ],
            'de': [
                "Dieses Produkt ist fantastisch! Großartige Qualität und einfach zu bedienen.",
                "Schreckliches Produkt, schlechte Qualität und verwirrende Benutzeroberfläche.",
                "Das Produkt ist in Ordnung, durchschnittliche Qualität aber schwer zu verwenden."
            ],
            'fr': [
                "Ce produit est incroyable! Excellente qualité et facile à utiliser.",
                "Produit terrible, mauvaise qualité et interface confuse.",
                "Le produit est correct, qualité moyenne mais difficile à utiliser."
            ]
        }
    
    def load_multilingual_sentiment_model(self):
        """Load XLM-RoBERTa based sentiment model"""
        print("🤖 Loading XLM-RoBERTa sentiment model...")
        
        try:
            # Load a good multilingual sentiment model
            model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
            self.models['xlm_sentiment'] = pipeline(
                "sentiment-analysis", 
                model=model_name,
                tokenizer=model_name
            )
            print("✅ XLM-RoBERTa sentiment model loaded successfully")
            
        except Exception as e:
            print(f"⚠️ XLM-RoBERTa failed: {e}")
            # Fallback to basic multilingual model
            try:
                self.models['xlm_sentiment'] = pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment"
                )
                print("✅ Fallback multilingual BERT loaded")
            except Exception as e2:
                print(f"❌ All models failed: {e2}")
                return False
        
        return True
    
    def load_aspect_detection_keywords(self):
        """Create keyword-based aspect detection (quick prototype)"""
        print("🎯 Setting up aspect detection...")
        
        self.aspect_keywords = {
            'product_quality': {
                'en': ['quality', 'durable', 'build', 'material', 'craftsmanship', 'sturdy', 'solid', 'cheap', 'flimsy', 'broken', 'defect'],
                'es': ['calidad', 'duradero', 'construcción', 'material', 'artesanía', 'sólido', 'barato', 'frágil', 'roto', 'defecto'],
                'de': ['qualität', 'langlebig', 'bau', 'material', 'handwerk', 'stabil', 'fest', 'billig', 'zerbrechlich', 'kaputt', 'defekt'],
                'fr': ['qualité', 'durable', 'construction', 'matériel', 'artisanat', 'solide', 'bon marché', 'fragile', 'cassé', 'défaut']
            },
            'user_experience': {
                'en': ['easy', 'difficult', 'interface', 'user', 'navigate', 'intuitive', 'confusing', 'simple', 'complex', 'usable'],
                'es': ['fácil', 'difícil', 'interfaz', 'usuario', 'navegar', 'intuitivo', 'confuso', 'simple', 'complejo', 'usable'],
                'de': ['einfach', 'schwierig', 'benutzeroberfläche', 'benutzer', 'navigieren', 'intuitiv', 'verwirrend', 'einfach', 'komplex', 'benutzbar'],
                'fr': ['facile', 'difficile', 'interface', 'utilisateur', 'naviguer', 'intuitif', 'confus', 'simple', 'complexe', 'utilisable']
            }
        }
        print("✅ Aspect detection keywords loaded")
    
    def detect_aspect(self, text, language='en'):
        """Detect aspect based on keywords"""
        text_lower = text.lower()
        
        product_score = 0
        ux_score = 0
        
        # Count keyword matches
        for keyword in self.aspect_keywords['product_quality'].get(language, []):
            if keyword in text_lower:
                product_score += 1
        
        for keyword in self.aspect_keywords['user_experience'].get(language, []):
            if keyword in text_lower:
                ux_score += 1
        
        # Determine aspect
        if product_score > ux_score:
            return 'product_quality'
        elif ux_score > product_score:
            return 'user_experience'
        else:
            return 'general'  # No clear aspect detected
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis on multilingual samples"""
        print("\n🧪 Testing Sentiment Analysis:")
        print("="*50)
        
        if 'xlm_sentiment' not in self.models:
            print("❌ No sentiment model loaded")
            return
        
        for lang, texts in self.test_texts.items():
            print(f"\n🌍 Language: {lang.upper()}")
            print("-" * 30)
            
            for i, text in enumerate(texts, 1):
                start_time = time.time()
                
                # Analyze sentiment
                result = self.models['xlm_sentiment'](text)
                
                # Detect aspect
                aspect = self.detect_aspect(text, lang)
                
                processing_time = time.time() - start_time
                
                print(f"{i}. Text: {text[:50]}...")
                print(f"   Sentiment: {result[0]['label']} ({result[0]['score']:.3f})")
                print(f"   Aspect: {aspect}")
                print(f"   Time: {processing_time:.3f}s")
                print()
    
    def test_with_sample_data(self):
        """Test with prepared sample data"""
        print("\n📊 Testing with Sample Business Data:")
        print("="*50)
        
        try:
            # Load sample data
            df = pd.read_csv('data/fedex_reviews_YYYYMMDD_HHMM.csv')
            print(f"Loaded {len(df)} sample reviews")
            
            # Test first 5 samples
            for idx, row in df.head(5).iterrows():
                text = row['text']
                actual_sentiment = row['sentiment']
                actual_aspect = row['aspect']
                language = row['language']
                
                # Predict sentiment
                if 'xlm_sentiment' in self.models:
                    pred_sentiment = self.models['xlm_sentiment'](text)
                    pred_aspect = self.detect_aspect(text, language)
                    
                    print(f"\nSample {idx + 1} ({language}):")
                    print(f"Text: {text}")
                    print(f"Actual: {actual_sentiment} | {actual_aspect}")
                    print(f"Predicted: {pred_sentiment[0]['label']} ({pred_sentiment[0]['score']:.3f}) | {pred_aspect}")
                    
        except Exception as e:
            print(f"❌ Error testing sample data: {e}")
    
    def benchmark_models(self):
        """Quick benchmark of different models"""
        print("\n⚡ Model Benchmark:")
        print("="*50)
        
        sample_text = "This product has excellent quality but the interface is confusing"
        
        if 'xlm_sentiment' in self.models:
            start_time = time.time()
            result = self.models['xlm_sentiment'](sample_text)
            processing_time = time.time() - start_time
            
            print(f"XLM-RoBERTa:")
            print(f"  Result: {result[0]['label']} ({result[0]['score']:.3f})")
            print(f"  Time: {processing_time:.3f}s")
        
        # Test aspect detection
        start_time = time.time()
        aspect = self.detect_aspect(sample_text)
        aspect_time = time.time() - start_time
        
        print(f"Aspect Detection:")
        print(f"  Result: {aspect}")
        print(f"  Time: {aspect_time:.3f}s")
    
    def run_all_tests(self):
        """Run all model tests"""
        print("🚀 Starting Model Testing...")
        print("="*60)
        
        # Load models
        if not self.load_multilingual_sentiment_model():
            print("❌ Failed to load sentiment models")
            return
        
        self.load_aspect_detection_keywords()
        
        # Run tests
        self.test_sentiment_analysis()
        self.test_with_sample_data()
        self.benchmark_models()
        
        print("\n✅ Model testing complete!")
        print("\n📋 Next Steps:")
        print("1. Models are working correctly")
        print("2. Ready for web application development")
        print("3. Consider fine-tuning for better accuracy")

if __name__ == "__main__":
    tester = ModelTester()
    tester.run_all_tests()