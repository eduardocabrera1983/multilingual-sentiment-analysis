#!/usr/bin/env python3
"""
Debug script to check which file the dashboard is loading and why only 49 reviews
Run this to identify the issue
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime

# Your paths
project_root = Path(r"D:\ironhack\Coursework\final_module_project\multilingual-sentiment-analysis")
data_dir = project_root / 'data'
cache_dir = project_root / 'cache'

print("DEBUGGING DASHBOARD DATA LOADING")
print("="*60)

# Replicate the dashboard's file search logic
data_sources = []

for directory in [data_dir, cache_dir]:
    if directory.exists():
        print(f"\nSearching in: {directory}")
        
        # Same order as your app.py
        patterns = [
            ('fedex_reviews_enhanced_ensemble_*.csv', 'FedEx Real Data'),
            ('fedex_reviews_*.csv', 'FedEx Data'),
            ('results_ensemble_*.csv', 'Ensemble Results'),
            ('results_*.csv', 'Analysis Results')
        ]
        
        for pattern, source_name in patterns:
            files = list(directory.glob(pattern))
            for f in files:
                if f.exists():
                    data_sources.append((f, source_name))
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    size_kb = f.stat().st_size / 1024
                    print(f"  Found: {f.name}")
                    print(f"    Type: {source_name}")
                    print(f"    Modified: {mtime}")
                    print(f"    Size: {size_kb:.1f} KB")

# Sort by modification time (newest first) - same as your app
data_sources.sort(key=lambda x: x[0].stat().st_mtime if x[0].exists() else 0, reverse=True)

if not data_sources:
    print("\nNO FILES FOUND - Dashboard would run in demo mode")
    exit()

print(f"\nFILE PRIORITY ORDER:")
print("="*40)

for i, (data_path, source_name) in enumerate(data_sources, 1):
    marker = ">>> DASHBOARD WILL LOAD THIS <<<" if i == 1 else "    backup option"
    print(f"{i}. {marker}")
    print(f"   File: {data_path.name}")
    print(f"   Type: {source_name}")
    print()

# Load the file that dashboard would load
top_file, source_name = data_sources[0]
print(f"LOADING TOP PRIORITY FILE: {top_file.name}")
print("="*60)

try:
    df = pd.read_csv(top_file)
    print(f"✓ Successfully loaded {len(df)} rows")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Check data quality - replicate dashboard logic
    print(f"\nDATA QUALITY CHECK:")
    print("="*30)
    
    # Check for required columns
    required_cols = ['sentiment', 'primary_aspect', 'classification_type', 'priority_level']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        print("   Dashboard might skip this file!")
    else:
        print(f"✓ All required columns present")
    
    # Check for empty/null data that might cause filtering
    print(f"\nNULL DATA CHECK:")
    for col in required_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            empty_count = (df[col] == '').sum() if df[col].dtype == 'object' else 0
            total_bad = null_count + empty_count
            if total_bad > 0:
                print(f"  {col}: {total_bad} null/empty values ({total_bad/len(df)*100:.1f}%)")
            else:
                print(f"  {col}: ✓ no missing data")
    
    # Show sample of data that dashboard would process
    print(f"\nSAMPLE DATA (first 5 rows):")
    print("="*40)
    display_cols = ['sentiment', 'primary_aspect', 'classification_type', 'priority_level']
    available_cols = [col for col in display_cols if col in df.columns]
    if available_cols:
        print(df[available_cols].head().to_string(index=False))
    else:
        print("Cannot display sample - missing expected columns")
    
    # Test dashboard data generation logic
    print(f"\nTESTING DASHBOARD METRICS GENERATION:")
    print("="*45)
    
    # Replicate generate_dashboard_data function
    data = {
        'total_reviews': len(df),
        'mixed_concerns_pct': '0',
        'ux_priority_pct': '0', 
        'high_priority_pct': '0',
        'sentiment_distribution': {'positive': 33, 'negative': 33, 'neutral': 34}
    }
    
    # Update with actual data if columns exist
    if 'classification_type' in df.columns:
        mixed = (df['classification_type'] == 'mixed_concerns').sum()
        mixed_pct = (mixed / len(df)) * 100
        data['mixed_concerns_pct'] = f"{mixed_pct:.1f}"
        print(f"Mixed concerns: {mixed} reviews ({mixed_pct:.1f}%)")
    
    if 'primary_aspect' in df.columns:
        ux = (df['primary_aspect'] == 'user_experience').sum()
        ux_pct = (ux / len(df)) * 100
        data['ux_priority_pct'] = f"{ux_pct:.1f}"
        print(f"UX priority: {ux} reviews ({ux_pct:.1f}%)")
    
    if 'priority_level' in df.columns:
        high = (df['priority_level'] == 'HIGH').sum()
        high_pct = (high / len(df)) * 100
        data['high_priority_pct'] = f"{high_pct:.1f}"
        print(f"High priority: {high} reviews ({high_pct:.1f}%)")
    
    if 'sentiment' in df.columns:
        sentiment_counts = df['sentiment'].value_counts()
        total = len(df)
        print(f"Sentiment distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"  {sentiment}: {count} reviews ({count/total*100:.1f}%)")
    
    print(f"\nEXPECTED DASHBOARD METRICS:")
    print(f"  Total Reviews: {data['total_reviews']}")
    print(f"  Mixed Concerns: {data['mixed_concerns_pct']}%")
    print(f"  UX Priority: {data['ux_priority_pct']}%") 
    print(f"  High Priority: {data['high_priority_pct']}%")
    
    # Compare with what dashboard shows
    print(f"\nDASHBOARD COMPARISON:")
    print("="*30)
    print("Dashboard shows 49 reviews, but file contains", len(df))
    
    if len(df) != 49:
        print(f"❌ MISMATCH: File has {len(df)} reviews but dashboard shows 49")
        print("\nPOSSIBLE CAUSES:")
        print("1. Dashboard is loading a different file")
        print("2. Data is being filtered due to missing/invalid values")
        print("3. There's another file with 49 reviews that has higher priority")
        print("4. Dashboard cache needs to be refreshed")
        
        # Check if there's a file with 49 rows
        print(f"\nSEARCHING FOR FILES WITH 49 ROWS:")
        for check_file, _ in data_sources:
            try:
                check_df = pd.read_csv(check_file)
                if len(check_df) == 49:
                    print(f"  ✓ FOUND: {check_file.name} has exactly 49 rows!")
                    print(f"    This might be what the dashboard is loading")
                else:
                    print(f"    {check_file.name}: {len(check_df)} rows")
            except:
                print(f"    {check_file.name}: Could not read")
    else:
        print("✓ File size matches dashboard display")

except Exception as e:
    print(f"❌ Error loading file: {e}")

print(f"\nRECOMMENDATIONS:")
print("="*20)
print("1. Check Flask console logs for which file is actually being loaded")
print("2. Restart Flask app to refresh file detection")
print("3. Use browser dev tools to check dashboard API calls")
print("4. Manually refresh dashboard data using the Refresh button")
print("5. Check if there are multiple CSV files and ensure the right one has priority")