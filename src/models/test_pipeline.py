#!/usr/bin/env python3
"""
Test script for Enhanced ML Pipeline - Uses full ensemble models
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

def test_enhanced_pipeline():
    print("🚀 Testing Enhanced ML Pipeline with Full Ensemble")
    print("="*60)
    
    try:
        # Try to import enhanced models directly
        from src.models.enhanced_sentiment_classifier import EnhancedSentimentClassifier
        from src.models.enhanced_aspect_classifier import EnhancedAspectClassifier
        
        print("✅ Enhanced models imported successfully!")
        
        # Initialize enhanced classifiers directly
        print("\n🤖 Initializing Enhanced Sentiment Classifier...")
        sentiment_classifier = EnhancedSentimentClassifier(use_ensemble=True)
        
        print("🎯 Initializing Enhanced Aspect Classifier...")
        aspect_classifier = EnhancedAspectClassifier(confidence_threshold=0.3)
        
        print("✅ Full enhanced system loaded!")
        
    except Exception as e:
        print(f"❌ Failed to load enhanced models: {e}")
        print("📋 Falling back to integrated pipeline...")
        
        try:
            from src.integrated_ml_pipeline import IntegratedMLPipeline
            pipeline = IntegratedMLPipeline()
            print("✅ Fallback pipeline loaded")
            
            # Test with fallback
            test_with_fallback(pipeline)
            return
            
        except Exception as e2:
            print(f"❌ Even fallback failed: {e2}")
            return
    
    # Test with enhanced models
    test_with_enhanced_models(sentiment_classifier, aspect_classifier)

def test_with_enhanced_models(sentiment_classifier, aspect_classifier):
    """Test with full enhanced models"""
    
    test_texts = [
        "This product has excellent quality but the interface is confusing",
        "App crashes frequently and the user experience is terrible", 
        "Very easy to use and great build quality",
        "La calidad del producto es excelente pero la interfaz es confusa"
    ]
    
    print(f"\n🧪 Testing Enhanced Models on {len(test_texts)} texts:")
    print("="*60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: {text}")
        
        # Analyze sentiment with enhanced model
        sentiment_result = sentiment_classifier.analyze_sentiment(text)
        
        # Analyze aspect with enhanced model  
        aspect_result = aspect_classifier.classify_aspect(text)
        
        print(f"   🤖 Enhanced Results:")
        print(f"      Sentiment: {sentiment_result['sentiment']} ({sentiment_result['confidence']:.3f})")
        print(f"      Models used: {sentiment_result.get('models_available', 'N/A')}")
        print(f"      Aspect: {aspect_result['aspect']} ({aspect_result['confidence']:.3f})")
        print(f"      Method: {aspect_result.get('method', 'N/A')}")
        print(f"      Processing: {sentiment_result['processing_time']:.3f}s")
        
        # Show model ensemble info
        if 'model_used' in sentiment_result:
            print(f"      Model ensemble: {sentiment_result['model_used']}")

def test_with_fallback(pipeline):
    """Test with fallback pipeline"""
    
    test_text = "This product has excellent quality but the interface is confusing"
    
    print(f"\n🧪 Testing Fallback Pipeline:")
    print(f"Text: {test_text}")
    
    try:
        result = pipeline.analyze_text(test_text)
        
        print(f"\n📊 Fallback Results:")
        print(f"   Sentiment: {result['sentiment']} (confidence: {result['sentiment_confidence']:.3f})")
        print(f"   Aspect: {result['aspect']} (confidence: {result['aspect_confidence']:.3f})")
        print(f"   Processing time: {result['processing_time']:.3f} seconds")
        print(f"   Language: {result['language']}")
        
        print(f"\n⚠️ Note: Using fallback models, not full enhanced system")
        
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")

if __name__ == "__main__":
    test_enhanced_pipeline()