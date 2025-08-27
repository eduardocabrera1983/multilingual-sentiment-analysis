import os
from pathlib import Path

project_root = Path(__file__).parent

print("Checking Flask app setup...")
print("=" * 50)

# Check directories
dirs_to_check = ['templates', 'static/css', 'static/js', 'uploads', 'cache', 'src/models', 'data']
for dir_path in dirs_to_check:
    full_path = project_root / dir_path
    exists = full_path.exists()
    print(f"{'✅' if exists else '❌'} {dir_path}: {'Found' if exists else 'Missing'}")

print("\nChecking template files...")
print("-" * 50)

templates = ['base.html', 'index.html', 'analyze.html', 'results.html', 'upload.html', 
             'batch_results.html', 'dashboard.html', 'about.html', '404.html', '500.html']

for template in templates:
    path = project_root / 'templates' / template
    exists = path.exists()
    print(f"{'✅' if exists else '❌'} {template}: {'Found' if exists else 'Missing'}")

print("\nChecking model files...")
print("-" * 50)

models = [
    'src/models/enhanced_sentiment_classifier.py',
    'src/models/enhanced_aspect_classifier.py',
    'src/pipelines/integrated_ml_pipeline.py'
]

# Also check alternative path
if not (project_root / 'src/pipelines/integrated_ml_pipeline.py').exists():
    models.append('src/integrated_ml_pipeline.py')

for model in models:
    path = project_root / model
    exists = path.exists()
    print(f"{'✅' if exists else '❌'} {model}: {'Found' if exists else 'Missing'}")

print("\nChecking data files...")
print("-" * 50)

data_files = list((project_root / 'data').glob('*.csv')) if (project_root / 'data').exists() else []
print(f"Found {len(data_files)} CSV files in data folder")
for file in data_files[:5]:  # Show first 5
    print(f"  - {file.name}")

print("\n" + "=" * 50)
print("Setup check complete!")