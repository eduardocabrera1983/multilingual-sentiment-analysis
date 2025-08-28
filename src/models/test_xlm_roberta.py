#!/usr/bin/env python3
"""
Step 1: Test XLM-RoBERTa single model
Run this to verify your GPU can handle the primary model
"""

import torch
from transformers import pipeline
import time

def test_xlm_roberta():
    print("Testing XLM-RoBERTa Primary Model...")
    print("=" * 50)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total VRAM: {total_memory:.1f} GB")
        device = 0
    else:
        print("Using CPU")
        device = -1
    
    try:
        # Clear any existing GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load XLM-RoBERTa
        start_time = time.time()
        model = pipeline(
            'sentiment-analysis',
            model='cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual',
            device=device,
            return_all_scores=True,
            truncation=True,
            max_length=512
        )
        load_time = time.time() - start_time
        
        print(f"✅ XLM-RoBERTa loaded in {load_time:.1f} seconds")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU Memory Used: {allocated:.2f} GB")
            print(f"Remaining: {total_memory - allocated:.2f} GB")
        
        # Test with your sample texts
        test_texts = [
            "not receiving email for sign in, this app continues to be trash!",
            "The app works so good I want to recommend it to all my colleagues.",
            "App crashes constantly and interface is terrible"
        ]
        
        print("\nTesting predictions:")
        print("-" * 30)
        
        for text in test_texts:
            start = time.time()
            result = model(text)
            pred_time = time.time() - start
            
            print(f"Text: {text[:50]}...")
            print(f"Result: {result}")
            print(f"Time: {pred_time*1000:.1f}ms\n")
        
        print("✅ XLM-RoBERTa working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_xlm_roberta()
    if success:
        print("\n🎯 Ready for Phase 2: Add mBERT")
    else:
        print("\n⚠️ Fix XLM-RoBERTa issues before proceeding")