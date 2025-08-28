#!/usr/bin/env python3
"""
Debug script to identify why dashboard shows random data instead of your 1000 FedEx reviews
Run this in your project directory to diagnose the issue
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# Your project paths (update these if different)
project_root = Path.cwd()  # Current directory
data_dir = project_root / 'data'
cache_dir = project_root / 'cache'

print("🔍 DEBUGGING DASHBOARD DATA LOADING")
print("=" * 60)
print(f"Project root: {project_root}")
print(f"Data directory: {data_dir}")
print(f"Cache directory: {cache_dir}")

def check_directories():
    """Check if required directories exist"""
    print(f"\n📂 DIRECTORY CHECK:")
    print("-" * 30)
    
    for name, path in [("Data", data_dir), ("Cache", cache_dir)]:
        if path.exists():
            file_count = len(list(path.glob("*.csv")))
            print(f"✅ {name}: {path} (contains {file_count} CSV files)")
        else:
            print(f"❌ {name}: {path} (MISSING)")
            
def find_all_csv_files():
    """Find all CSV files that could be dashboard data sources"""
    print(f"\n🔍 SEARCHING FOR CSV FILES:")
    print("-" * 40)
    
    data_sources = []
    
    # Search patterns matching your app.py logic
    search_locations = [
        (data_dir, "Data Directory"),
        (cache_dir, "Cache Directory")
    ]
    
    patterns = [
        ('fedex_reviews_enhanced_ensemble_*.csv', 'FedEx Real Data'),
        ('fedex_reviews_*.csv', 'FedEx Data'), 
        ('results_ensemble_*.csv', 'Ensemble Results'),
        ('results_*.csv', 'Analysis Results')
    ]
    
    for directory, dir_name in search_locations:
        if not directory.exists():
            print(f"⚠️  {dir_name} does not exist: {directory}")
            continue
            
        print(f"\n📁 {dir_name}: {directory}")
        found_files = False
        
        for pattern, source_type in patterns:
            files = list(directory.glob(pattern))
            
            for file_path in files:
                if file_path.exists():
                    found_files = True
                    mtime = file_path.stat().st_mtime
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    
                    data_sources.append((file_path, source_type, mtime))
                    
                    print(f"  ✅ {file_path.name}")
                    print(f"     Type: {source_type}")
                    print(f"     Modified: {datetime.fromtimestamp(mtime)}")
                    print(f"     Size: {size_mb:.2f} MB")
        
        if not found_files:
            print(f"  ⚠️  No CSV files found matching expected patterns")
    
    return data_sources

def analyze_file_priority(data_sources):
    """Determine which file dashboard would load (newest first)"""
    print(f"\n🏆 FILE PRIORITY (Dashboard loads first match):")
    print("-" * 50)
    
    if not data_sources:
        print("❌ NO FILES FOUND - Dashboard will use DEMO MODE")
        return None
        
    # Sort by modification time (newest first) - same as app.py
    data_sources.sort(key=lambda x: x[2], reverse=True)
    
    for i, (file_path, source_type, mtime) in enumerate(data_sources, 1):
        indicator = "🎯 DASHBOARD WILL LOAD THIS" if i == 1 else "   backup option"
        file_time = datetime.fromtimestamp(mtime)
        
        print(f"{i:2d}. {indicator}")
        print(f"    📄 {file_path.name}")
        print(f"    📊 {source_type}")
        print(f"    📅 {file_time}")
        print()
    
    return data_sources[0] if data_sources else None

def test_file_loading(file_info):
    """Test loading and processing the priority file"""
    if not file_info:
        return
        
    file_path, source_type, mtime = file_info
    print(f"🧪 TESTING FILE LOADING: {file_path.name}")
    print("-" * 50)
    
    try:
        # Load file same way as dashboard
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"✅ Successfully loaded: {len(df):,} rows, {len(df.columns)} columns")
        
        # Test dashboard validation logic
        print(f"\n🔍 DASHBOARD VALIDATION CHECKS:")
        
        # Check 1: Minimum columns
        if len(df.columns) < 5:
            print(f"❌ FAIL: Too few columns ({len(df.columns)} < 5)")
            print("   Dashboard will skip this file!")
            return
        else:
            print(f"✅ PASS: Sufficient columns ({len(df.columns)} >= 5)")
        
        # Check 2: Required columns
        expected_cols = ['sentiment', 'primary_aspect', 'classification_type', 'priority_level']
        has_predicted_cols = any(col.startswith('predicted_') for col in df.columns)
        has_original_cols = any(col in df.columns for col in expected_cols)
        
        if not (has_original_cols or has_predicted_cols):
            print(f"❌ FAIL: Missing expected columns")
            print(f"   Expected: {expected_cols}")
            print(f"   Available: {list(df.columns)[:10]}...")
            print("   Dashboard will skip this file!")
            return
        else:
            if has_original_cols:
                print(f"✅ PASS: Has original format columns")
                found_cols = [col for col in expected_cols if col in df.columns]
                print(f"   Found: {found_cols}")
            if has_predicted_cols:
                print(f"✅ PASS: Has predicted format columns")
        
        # Check 3: Empty dataframe
        if len(df) == 0:
            print(f"❌ FAIL: Empty dataframe")
            return
        else:
            print(f"✅ PASS: Non-empty dataframe ({len(df):,} rows)")
            
        # Generate dashboard metrics
        print(f"\n📊 TESTING DASHBOARD METRICS GENERATION:")
        print("-" * 40)
        
        metrics = generate_test_dashboard_data(df)
        
        print(f"Total Reviews: {metrics['total_reviews']:,}")
        print(f"Mixed Concerns: {metrics['mixed_concerns_pct']}%")
        print(f"UX Priority: {metrics['ux_priority_pct']}%")
        print(f"High Priority: {metrics['high_priority_pct']}%")
        print(f"Sentiment Distribution:")
        for sentiment, pct in metrics['sentiment_distribution'].items():
            print(f"  {sentiment}: {pct}%")
            
        # Show sample data
        print(f"\n📋 SAMPLE DATA (first 3 rows):")
        display_cols = [col for col in ['sentiment', 'primary_aspect', 'classification_type', 'priority_level'] if col in df.columns]
        if display_cols:
            print(df[display_cols].head(3).to_string(index=False))
        
    except Exception as e:
        print(f"❌ ERROR loading file: {e}")
        print("Dashboard will skip this file and continue to next option")

def generate_test_dashboard_data(df):
    """Replicate the generate_dashboard_data function from app.py"""
    data = {
        'total_reviews': len(df),
        'mixed_concerns_pct': '0',
        'ux_priority_pct': '0', 
        'high_priority_pct': '0',
        'sentiment_distribution': {'positive': 33, 'negative': 33, 'neutral': 34}
    }
    
    # Test both column formats (predicted_ and direct)
    classification_col = 'predicted_classification_type' if 'predicted_classification_type' in df.columns else 'classification_type'
    if classification_col in df.columns:
        mixed = (df[classification_col] == 'mixed_concerns').mean() * 100
        data['mixed_concerns_pct'] = f"{mixed:.1f}"
    
    aspect_col = 'predicted_primary_aspect' if 'predicted_primary_aspect' in df.columns else 'primary_aspect'
    if aspect_col in df.columns:
        ux = (df[aspect_col] == 'user_experience').mean() * 100
        data['ux_priority_pct'] = f"{ux:.1f}"
    
    priority_col = 'predicted_priority_level' if 'predicted_priority_level' in df.columns else 'priority_level'
    if priority_col in df.columns:
        high = (df[priority_col] == 'HIGH').mean() * 100
        data['high_priority_pct'] = f"{high:.1f}"
    
    sentiment_col = 'predicted_sentiment' if 'predicted_sentiment' in df.columns else 'sentiment'
    if sentiment_col in df.columns:
        sentiments = df[sentiment_col].value_counts(normalize=True) * 100
        data['sentiment_distribution'] = {
            'positive': int(sentiments.get('positive', 0)),
            'negative': int(sentiments.get('negative', 0)),
            'neutral': int(sentiments.get('neutral', 0))
        }
    
    return data

def provide_solutions():
    """Provide step-by-step solutions"""
    print(f"\n💡 SOLUTIONS:")
    print("=" * 20)
    
    print("1. 🔄 IMMEDIATE FIX - Restart Flask App:")
    print("   - Stop your Flask app (Ctrl+C)")
    print("   - Restart: python app.py")
    print("   - Check console logs for file loading messages")
    print()
    
    print("2. 🔍 VERIFY FILE LOCATION:")
    print("   - Ensure your FedEx file is in the 'data' directory")
    print("   - File should match pattern: fedex_reviews_enhanced_ensemble_*.csv")
    print()
    
    print("3. 🧹 CLEAR CACHE:")
    print("   - Delete any old files in cache/ directory")
    print("   - Ensure your FedEx file is the newest (most recent modification time)")
    print()
    
    print("4. 🔧 MANUAL DASHBOARD REFRESH:")
    print("   - Open dashboard in browser")
    print("   - Click the 'Refresh' button") 
    print("   - Check browser console (F12) for errors")
    print()
    
    print("5. 📊 VERIFY IN BROWSER:")
    print("   - Open: http://localhost:5000/api/dashboard/data")
    print("   - Should show your actual data, not demo mode")

def main():
    """Run complete diagnostic"""
    check_directories()
    data_sources = find_all_csv_files()
    priority_file = analyze_file_priority(data_sources)
    test_file_loading(priority_file)
    provide_solutions()
    
    if not data_sources:
        print(f"\n⚠️  CRITICAL: No data files found!")
        print("Make sure your FedEx CSV file is in the correct location.")

if __name__ == "__main__":
    main()