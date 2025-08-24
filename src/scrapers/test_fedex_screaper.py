#!/usr/bin/env python3
"""
Diagnostic Script for FedEx Scraper Model Loading Issues
Run this to identify exactly where the problem occurs
"""

import time
import torch
import sys
import gc
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def diagnostic_check():
    """Run comprehensive diagnostic checks"""
    
    print("🔍 FEDEX SCRAPER DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # 1. System Check
    print("\n1️⃣ SYSTEM CHECK")
    print(f"   Python Version: {sys.version}")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   Current Usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # 2. Network Check
    print("\n2️⃣ NETWORK CONNECTIVITY CHECK")
    try:
        import requests
        response = requests.get("https://huggingface.co/", timeout=10)
        print(f"   HuggingFace Hub: ✅ Accessible ({response.status_code})")
    except Exception as e:
        print(f"   HuggingFace Hub: ❌ Error - {e}")
        print("   This could cause model download issues!")
    
    # 3. Model Loading Test
    print("\n3️⃣ MODEL LOADING TEST")
    
    models_to_test = [
        {
            'name': 'DistilBERT (Working)',
            'id': 'lxyuan/distilbert-base-multilingual-cased-sentiments-student',
            'expected_time': 30
        },
        {
            'name': 'XLM-RoBERTa (Problematic)', 
            'id': 'cardiffnlp/twitter-xlm-roberta-base-sentiment',
            'expected_time': 60
        }
    ]
    
    for model_info in models_to_test:
        print(f"\n   Testing {model_info['name']}...")
        print(f"   Model ID: {model_info['id']}")
        
        # Test with timeout
        start_time = time.time()
        timeout = model_info['expected_time']
        success = False
        
        try:
            print(f"   ⏳ Loading (timeout: {timeout}s)...")
            
            # First try: Check if model is cached
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_info['id'], 
                    local_files_only=True
                )
                print("   📝 Tokenizer found in cache")
                
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_info['id'],
                    local_files_only=True
                )
                print("   🧠 Model found in cache")
                
            except Exception:
                print("   📥 Model not cached, downloading...")
                
                # Download with timeout simulation
                import threading
                
                result = {'model': None, 'error': None}
                
                def load_model():
                    try:
                        result['model'] = pipeline(
                            'sentiment-analysis',
                            model=model_info['id'],
                            device=-1  # Use CPU for diagnostic
                        )
                    except Exception as e:
                        result['error'] = str(e)
                
                # Start loading in thread
                thread = threading.Thread(target=load_model)
                thread.daemon = True
                thread.start()
                
                # Wait with timeout
                thread.join(timeout=timeout)
                
                if thread.is_alive():
                    print(f"   ⏰ TIMEOUT after {timeout}s - This is your problem!")
                    print("   💡 Model download is hanging")
                    print("   💡 Try: Download models manually first")
                    continue
                
                if result['error']:
                    print(f"   ❌ Error: {result['error']}")
                    continue
                
                if result['model']:
                    print("   ✅ Model loaded successfully")
                    
                    # Test inference
                    test_result = result['model']("This is a test")
                    print(f"   🧪 Test inference: {test_result}")
            
            load_time = time.time() - start_time
            print(f"   ✅ {model_info['name']} OK ({load_time:.1f}s)")
            success = True
            
        except Exception as e:
            print(f"   ❌ {model_info['name']} FAILED: {str(e)[:100]}")
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # 4. Recommendations
    print("\n4️⃣ RECOMMENDATIONS")
    print("   Based on the diagnostic results above:")
    print("   ")
    print("   If XLM-RoBERTa timed out:")
    print("   → Run: python download_models.py")
    print("   → Or use single model mode")
    print("   ")
    print("   If network issues detected:")
    print("   → Check firewall/proxy settings")
    print("   → Try downloading models manually")
    print("   ")
    print("   If GPU memory issues:")
    print("   → Use CPU mode: force_cpu=True")
    print("   → Or reduce batch size")
    
    print(f"\n✅ Diagnostic complete!")

if __name__ == "__main__":
    diagnostic_check()