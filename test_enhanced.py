#!/usr/bin/env python3
"""
Simple test for enhanced ML models - save as test_enhanced.py in project root
"""

import sys
import os

# Add paths
sys.path.append('src')
sys.path.append('src/models')

def main():
    print("🚀 Direct Enhanced ML Test")
    print("="*40)
    
    try:
        # Import enhanced models
        from enhanced_sentiment_classifier import EnhancedSentimentClassifier
        from enhanced_aspect_classifier import EnhancedAspectClassifier
        
        print("✅ Enhanced models imported!")
        
        # Initialize
        print("\n🤖 Initializing enhanced models...")
        sentiment_classifier = EnhancedSentimentClassifier(use_ensemble=True)
        aspect_classifier = EnhancedAspectClassifier()
        
        # Test
        test_text = "This product has excellent quality but the interface is confusing"
        print(f"\n🧪 Testing: {test_text}")
        
        # Analyze
        sentiment_result = sentiment_classifier.analyze_sentiment(test_text)
        aspect_result = aspect_classifier.classify_aspect(test_text)
        
        # Results
        print(f"\n📊 Results:")
        print(f"   Sentiment: {sentiment_result['sentiment']} ({sentiment_result['confidence']:.3f})")
        print(f"   Aspect: {aspect_result['aspect']} ({aspect_result['confidence']:.3f})")
        print(f"   Time: {sentiment_result['processing_time']:.3f}s")
        
        if 'models_available' in sentiment_result:
            print(f"   Models: {sentiment_result['models_available']}")
        
        print(f"\n✅ Enhanced system working!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Debug info
        print(f"\nDebug info:")
        print(f"Current dir: {os.getcwd()}")
        print(f"Python path: {sys.path[:3]}")
        
        if os.path.exists('src/models'):
            files = os.listdir('src/models')
            print(f"Files in src/models: {files}")

if __name__ == "__main__":
    main()