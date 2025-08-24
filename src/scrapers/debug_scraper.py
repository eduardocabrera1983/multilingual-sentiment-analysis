# Create this debug script: debug_scraper.py
# This will help you find exactly where the NoneType error occurs

def debug_review_data():
    """Debug the review data to find None values"""
    try:
        from google_play_scraper import reviews, Sort
        
        # Get a small sample of reviews for debugging
        print("🔍 Debugging review data structure...")
        
        result, _ = reviews(
            'com.fedex.ida.android',  # FedEx app ID
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=5  # Just get 5 reviews for debugging
        )
        
        print(f"📦 Got {len(result)} reviews for debugging")
        
        # Examine each review's data structure
        for i, review in enumerate(result):
            print(f"\n🔍 Review {i+1}:")
            print(f"   Type: {type(review)}")
            
            if review is None:
                print("   ❌ This review is None!")
                continue
                
            # Check each field for None values
            for key, value in review.items():
                value_type = type(value).__name__
                is_none = value is None
                
                if is_none:
                    print(f"   ❌ {key}: None ({value_type}) - THIS WILL CAUSE .strip() ERROR")
                elif isinstance(value, str) and len(value) == 0:
                    print(f"   ⚠️  {key}: empty string ({value_type})")
                else:
                    preview = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    print(f"   ✅ {key}: {preview} ({value_type})")
                    
                # Test .strip() on this field
                if isinstance(value, str):
                    try:
                        _ = value.strip()
                        print(f"      ✅ .strip() works on {key}")
                    except Exception as e:
                        print(f"      ❌ .strip() FAILS on {key}: {e}")
                elif value is None:
                    print(f"      ❌ .strip() would FAIL on {key} (None value)")
        
        print("\n" + "="*60)
        print("🎯 DEBUGGING COMPLETE")
        print("Look for fields marked with ❌ - these are causing your error!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")

if __name__ == "__main__":
    debug_review_data()

# Run this debug script:
# python debug_scraper.py

# This will show you exactly which fields are None and causing the .strip() error