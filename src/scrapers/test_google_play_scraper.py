# Quick test to verify it works
from google_play_scraper import app, reviews

# Test with FedEx app
app_info = app('com.fedex.ida.android')
print(f"✅ Successfully fetched {app_info['title']}")
print(f"   Rating: {app_info['score']}")
print(f"   Installs: {app_info['installs']}")