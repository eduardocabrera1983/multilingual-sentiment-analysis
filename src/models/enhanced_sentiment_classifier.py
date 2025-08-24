import numpy as np
import pandas as pd
from transformers import pipeline
import torch
import warnings
from typing import Dict, List, Optional
import time
import logging
from datetime import datetime
import os

warnings.filterwarnings('ignore')

class EnhancedSentimentClassifier:
    """
    Fixed GPU-Accelerated Sentiment Classifier
    Optimized for NVIDIA RTX 4000 Ada (12GB VRAM)
    """
    
    def __init__(self, use_ensemble=True, device='auto', verbose=True):
        self.use_ensemble = use_ensemble
        self.verbose = verbose
        self.models = {}
        
        # GPU Configuration
        self.device = self._configure_gpu()
        
        print("\n" + "="*70)
        print("🚀 GPU-ACCELERATED SENTIMENT CLASSIFIER")
        print("="*70)
        print(f"🎮 Device: {self.device}")
        
        if self.device == 'cuda':
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print("⚡ Models will load 10-50x faster on GPU!")
        else:
            print("⚠️ Running on CPU (GPU not available)")
        
        print("="*70)
        
        # Load models
        self._load_models_gpu_optimized()
    
    def _configure_gpu(self):
        """Configure GPU with optimal settings"""
        if torch.cuda.is_available():
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Enable optimizations for Ada Lovelace architecture (RTX 4000)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            print("\n🎮 GPU Configuration:")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Available GPUs: {torch.cuda.device_count()}")
            
            return 'cuda'
        return 'cpu'
    
  
    def _load_models_gpu_optimized(self):
        """Load models optimized for GPU execution - FIXED FOR XLM-RoBERTa HANGING"""
        
        # Model configurations
        model_configs = [
            {
                'name': 'distilbert_multilingual',
                'model_id': 'lxyuan/distilbert-base-multilingual-cased-sentiments-student',
                'weight': 1.0,
                'primary': True
            },
            # {
            #     'name': 'xlm_roberta_base',
            #     'model_id': 'cardiffnlp/twitter-xlm-roberta-base-sentiment',
            #     'weight': 0.5,
            #     'primary': False
            # }
        ]
        
        if not self.use_ensemble:
            model_configs = [model_configs[0]]  # Use only primary model
        
        print(f"\n📦 Loading {len(model_configs)} model(s) on {self.device.upper()}...")
        
        successful_loads = 0
        
        for config in model_configs:
            try:
                print(f"\n⏳ Loading {config['name']}...")
                start_time = time.time()
                
                # FIXED: Progressive fallback approach for problematic XLM-RoBERTa
                if self.device == 'cuda':
                    model = None
                    loading_methods = [
                        # Method 1: Try FP16 first
                        lambda: pipeline(
                            'sentiment-analysis',
                            model=config['model_id'],
                            device=0,
                            torch_dtype=torch.float16
                        ),
                        # Method 2: Try FP32 if FP16 fails/hangs
                        lambda: pipeline(
                            'sentiment-analysis',
                            model=config['model_id'],
                            device=0
                            # No torch_dtype = uses default FP32
                        ),
                        # Method 3: CPU fallback
                        lambda: pipeline(
                            'sentiment-analysis',
                            model=config['model_id'],
                            device=-1
                        )
                    ]
                    
                    # Try each loading method with timeout
                    import threading
                    import queue
                    
                    for i, method in enumerate(loading_methods):
                        method_name = ['FP16', 'FP32', 'CPU'][i]
                        print(f"   🔄 Attempting {method_name} loading...")
                        
                        result_queue = queue.Queue()
                        
                        def load_with_method():
                            try:
                                model_result = method()
                                result_queue.put(('success', model_result))
                            except Exception as e:
                                result_queue.put(('error', str(e)))
                        
                        # Start loading in separate thread
                        thread = threading.Thread(target=load_with_method)
                        thread.daemon = True
                        thread.start()
                        
                        # Wait for result with timeout (30s for GPU, 60s for CPU)
                        timeout = 60 if method_name == 'CPU' else 30
                        thread.join(timeout=timeout)
                        
                        if thread.is_alive():
                            print(f"   ⏰ {method_name} timed out after {timeout}s")
                            continue  # Try next method
                        
                        try:
                            status, result = result_queue.get_nowait()
                            if status == 'success':
                                model = result
                                print(f"   ✅ Loaded with {method_name}")
                                break
                            else:
                                print(f"   ❌ {method_name} failed: {result[:50]}")
                        except queue.Empty:
                            print(f"   ❌ {method_name} failed (no result)")
                            continue
                    
                    if model is None:
                        print(f"   ⚠️ All methods failed for {config['name']}, skipping")
                        continue
                else:
                    # CPU mode (no issues here)
                    model = pipeline(
                        'sentiment-analysis',
                        model=config['model_id'],
                        device=-1
                    )
                
                load_time = time.time() - start_time
                
                self.models[config['name']] = {
                    'pipeline': model,
                    'weight': config['weight'],
                    'primary': config['primary'],
                    'model_id': config['model_id']
                }
                
                successful_loads += 1
                print(f"✅ {config['name']} loaded in {load_time:.1f}s")
                
                if self.device == 'cuda' and model.device != torch.device('cpu'):
                    # Show GPU memory usage
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"   GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
                
            except Exception as e:
                print(f"❌ Failed to load {config['name']}: {str(e)[:100]}")
                if config['primary'] and successful_loads == 0:
                    print("⚠️ Loading fallback model...")
                    self._load_fallback_model()
                    successful_loads += 1
        
        if successful_loads == 0:
            print("\n⚠️ All models failed, using lightweight fallback...")
            self._load_simple_fallback()
        
        # Normalize weights
        if self.models:
            total_weight = sum(m['weight'] for m in self.models.values())
            if total_weight > 0:
                for name in self.models:
                    self.models[name]['weight'] /= total_weight
        
        print(f"\n✅ Sentiment classifier ready with {len(self.models)} model(s)!")

    # Also add these helper methods if they don't exist:
    def _load_fallback_model(self):
        """Load a simple fallback model when main models fail"""
        try:
            model = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', device=-1)
            self.models['fallback'] = {
                'pipeline': model,
                'weight': 1.0,
                'primary': True,
                'model_id': 'distilbert-base-uncased-finetuned-sst-2-english'
            }
            print("✅ Fallback model loaded successfully")
        except Exception as e:
            print(f"❌ Even fallback model failed: {e}")

    def _load_simple_fallback(self):
        """Ultra-simple fallback using transformers' default sentiment pipeline"""
        try:
            model = pipeline('sentiment-analysis')  # Uses default model
            self.models['simple_fallback'] = {
                'pipeline': model,
                'weight': 1.0,
                'primary': True,
                'model_id': 'default'
            }
            print("✅ Simple fallback loaded")
        except Exception as e:
            print(f"❌ All fallbacks failed: {e}")
    
    def analyze_sentiment(self, text: str, language: str = 'auto') -> Dict:
        """Analyze sentiment using GPU acceleration"""
        if not text.strip():
            return self._neutral_result()
        
        start_time = time.time()
        
        # Try to use loaded models
        predictions = {}
        
        for name, model_info in self.models.items():
            try:
                if model_info['pipeline'] is not None:
                    # Use the model
                    with torch.no_grad():
                        result = model_info['pipeline'](text)
                        predictions[name] = result
                else:
                    # Rule-based fallback
                    predictions[name] = self._rule_based_analysis(text)
                    
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Model {name} failed: {str(e)[:50]}")
        
        # Process results
        if not predictions:
            return self._neutral_result()
        
        # Get first available result
        model_name = list(predictions.keys())[0]
        result = predictions[model_name]
        
        # Process the prediction
        if isinstance(result, dict) and 'sentiment' in result:
            # Rule-based result
            final_result = result
        else:
            # Model result
            final_result = self._process_model_prediction(result)
        
        processing_time = time.time() - start_time
        
        return {
            'sentiment': final_result['sentiment'],
            'confidence': final_result['confidence'],
            'scores': final_result['scores'],
            'language': language,
            'model_used': model_name,
            'processing_time': processing_time,
            'device': self.device
        }
    
    def _rule_based_analysis(self, text: str) -> Dict:
        """Simple rule-based sentiment analysis"""
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'trash']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return {
                'sentiment': 'positive',
                'confidence': 0.7,
                'scores': {'positive': 0.7, 'negative': 0.2, 'neutral': 0.1}
            }
        elif neg_count > pos_count:
            return {
                'sentiment': 'negative',
                'confidence': 0.7,
                'scores': {'positive': 0.2, 'negative': 0.7, 'neutral': 0.1}
            }
        else:
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'scores': {'positive': 0.3, 'negative': 0.3, 'neutral': 0.4}
            }
    
    def _process_model_prediction(self, prediction) -> Dict:
        """Process a model prediction to standard format"""
        if isinstance(prediction, list) and len(prediction) > 0:
            prediction = prediction[0]
        
        label = prediction.get('label', 'neutral').lower()
        score = prediction.get('score', 0.5)
        
        # Map labels to sentiments
        if 'pos' in label or label in ['positive', '5 stars', '4 stars']:
            sentiment = 'positive'
            scores = {'positive': score, 'negative': 1-score, 'neutral': 0.0}
        elif 'neg' in label or label in ['negative', '1 star', '2 stars']:
            sentiment = 'negative'
            scores = {'negative': score, 'positive': 1-score, 'neutral': 0.0}
        else:
            sentiment = 'neutral'
            scores = {'neutral': score, 'positive': (1-score)/2, 'negative': (1-score)/2}
        
        return {
            'sentiment': sentiment,
            'confidence': score,
            'scores': scores
        }
    
    def _neutral_result(self) -> Dict:
        """Return neutral result for empty text"""
        return {
            'sentiment': 'neutral',
            'confidence': 0.0,
            'scores': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33},
            'language': 'unknown',
            'model_used': 'none',
            'processing_time': 0.0,
            'device': self.device
        }
    
    def analyze_batch(self, texts: List[str], languages: List[str] = None) -> List[Dict]:
        """Analyze multiple texts efficiently"""
        if not texts:
            return []
        
        results = []
        batch_size = 32 if self.device == 'cuda' else 8
        
        print(f"\n⚡ Batch processing {len(texts)} texts on {self.device.upper()}...")
        start_time = time.time()
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            if len(texts) > 100 and i % 100 == 0:
                print(f"   Progress: {min(i + batch_size, len(texts))}/{len(texts)}")
            
            for text in batch:
                result = self.analyze_sentiment(text)
                results.append(result)
        
        total_time = time.time() - start_time
        print(f"✅ Processed {len(texts)} texts in {total_time:.1f}s")
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            'loaded_models': list(self.models.keys()),
            'ensemble_enabled': self.use_ensemble,
            'device': self.device,
            'model_count': len(self.models)
        }

# Test if running directly
if __name__ == "__main__":
    print("🔬 Testing Fixed GPU Sentiment Classifier")
    print("="*70)
    
    classifier = EnhancedSentimentClassifier(use_ensemble=False)
    
    test_texts = [
        "This app is amazing!",
        "Terrible experience",
        "It's okay",
        "not receiving email for sign in, this app continues to be trash!"
    ]
    
    for text in test_texts:
        result = classifier.analyze_sentiment(text)
        print(f"\n'{text}'")
        print(f"→ {result['sentiment'].upper()} ({result['confidence']:.2f})")
        print(f"  Device: {result['device']}, Time: {result['processing_time']*1000:.1f}ms")