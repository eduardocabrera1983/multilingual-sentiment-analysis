#!/usr/bin/env python3
"""
Fixed Test Script for FedEx Scraper
Handles proper path resolution for imports
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # Go up from tests to project root
sys.path.insert(0, str(project_root))

print(f"🔍 Project root: {project_root}")
print(f"🐍 Python path includes: {project_root}")

try:
    # Now try to import - this should work
    from src.scrapers.fedex_scraper import FedExReviewAnalyzer
    print("✅ Successfully imported FedExReviewAnalyzer")
    
    # Test basic initialization
    print("\n🚀 Testing FedEx Review Analyzer...")
    analyzer = FedExReviewAnalyzer(use_enhanced_models=False)  # Start with basic models
    print("✅ FedExReviewAnalyzer initialized successfully")
    
    # Test with small sample
    print("\n🧪 Running small test (5 reviews)...")
    try:
        df = analyzer.analyze_fedex_reviews(count=5, real_only=True)
        
        if df is not None and len(df) > 0:
            print(f"✅ Test successful! Collected {len(df)} reviews")
            print(f"📊 Columns: {list(df.columns)}")
            print(f"🌍 Countries: {df['country'].unique() if 'country' in df.columns else 'N/A'}")
            print(f"😊 Sentiments: {df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else 'N/A'}")
        else:
            print("⚠️ No reviews collected - this might be expected if google-play-scraper isn't installed")
            print("💡 Install it with: pip install google-play-scraper")
            
    except Exception as e:
        print(f"⚠️ Test run failed: {e}")
        print("💡 This is likely because google-play-scraper is not installed")
        print("💡 Install it with: pip install google-play-scraper")
    
    print("\n✅ Basic test completed!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    
    # Let's debug the path issue
    print("\n🔍 Debugging paths...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {current_file}")
    print(f"Project root: {project_root}")
    
    # Check if src directory exists
    src_path = project_root / 'src'
    scrapers_path = src_path / 'scrapers'
    fedex_file = scrapers_path / 'fedex_scraper.py'
    
    print(f"\n📁 Directory structure check:")
    print(f"   src/ exists: {src_path.exists()}")
    print(f"   src/scrapers/ exists: {scrapers_path.exists()}")
    print(f"   fedex_scraper.py exists: {fedex_file.exists()}")
    
    if not src_path.exists():
        print("\n❌ The 'src' directory doesn't exist!")
        print("💡 Make sure you're running from the correct project directory")
        print("💡 Your project structure should be:")
        print("   multilingual-sentiment-analysis/")
        print("   ├── src/")
        print("   │   ├── scrapers/")
        print("   │   │   └── fedex_scraper.py")
        print("   │   └── models/")
        print("   └── tests/")
        print("       └── test_scraper.py")
    
    if not fedex_file.exists():
        print(f"\n❌ fedex_scraper.py not found at: {fedex_file}")
        print("💡 Make sure you've saved the fixed file in the correct location")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("🏁 Test script completed")

# Alternative import method if the above fails
print("\n🔄 Trying alternative import method...")
try:
    # Try direct file import
    fedex_scraper_path = project_root / 'src' / 'scrapers' / 'fedex_scraper.py'
    if fedex_scraper_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("fedex_scraper", fedex_scraper_path)
        fedex_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fedex_module)
        
        FedExReviewAnalyzer = fedex_module.FedExReviewAnalyzer
        print("✅ Successfully imported via direct file method")
        
        # Quick test
        analyzer = FedExReviewAnalyzer(use_enhanced_models=False)
        print("✅ Direct import test successful!")
        
    else:
        print(f"❌ File not found: {fedex_scraper_path}")

except Exception as e:
    print(f"❌ Alternative import also failed: {e}")