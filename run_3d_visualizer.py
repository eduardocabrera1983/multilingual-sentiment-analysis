#!/usr/bin/env python3
"""
Simple runner script for 3D visualizations
"""

import sys
import os

# Install required packages if needed
def install_required_packages():
    """Install required packages for visualization"""
    required_packages = [
        'plotly>=5.0.0',
        'scikit-learn>=1.0.0', 
        'pandas>=1.3.0',
        'numpy>=1.20.0'
    ]
    
    print("📦 Checking required packages...")
    
    try:
        import plotly
        import sklearn
        import pandas
        import numpy
        print("✅ All packages already installed!")
        return True
    except ImportError as e:
        print(f"⚠️ Missing package: {e}")
        print("🔧 Installing required packages...")
        
        import subprocess
        for package in required_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
        
        print("✅ Packages installed successfully!")
        return True

def main():
    """Main execution function"""
    
    print("🚀 3D Sentiment Analysis Visualizer")
    print("="*50)
    
    # Check and install packages
    if not install_required_packages():
        print("❌ Package installation failed. Please install manually:")
        print("   pip install plotly scikit-learn pandas numpy")
        return
    
    # Import and run the visualizer
    try:
        # Import the 3D visualizer (assuming it's in the same directory)
        exec(open('sentiment_3d_visualizer.py').read())
        
    except FileNotFoundError:
        print("❌ Please save the 3D visualizer code as 'sentiment_3d_visualizer.py' first")
        print("💡 Copy the code from the artifact above into a file named 'sentiment_3d_visualizer.py'")
        
    except Exception as e:
        print(f"❌ Error running visualizer: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure you have the ML pipeline code in src/ directory")
        print("   2. Check that FedEx CSV data exists in data/ directory")
        print("   3. The script will create demo data if no CSV is found")

if __name__ == "__main__":
    main()