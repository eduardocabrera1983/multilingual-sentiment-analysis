# verify_install.py
import sys
import subprocess

venv_python = r"D:\ironhack\Coursework\final_module_project\multilingual-sentiment-analysis\.venv\Scripts\python.exe"

# Install using the venv Python directly
print("Installing google-play-scraper...")
subprocess.check_call([venv_python, "-m", "pip", "install", "google-play-scraper"])

# Test import
print("\nTesting import...")
subprocess.check_call([venv_python, "-c", "from google_play_scraper import app; print('✅ Success!')"])