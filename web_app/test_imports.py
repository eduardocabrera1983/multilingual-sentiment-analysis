echo "🧪 Testing model imports..."
python3 -c "
import sys
from pathlib import Path

# Add paths
project_root = Path.cwd()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

print(f'Project root: {project_root}')
print(f'Python path: {sys.path[:3]}...')

# Test imports
try:
    from src.models.enhanced_sentiment_classifier import EnhancedSentimentClassifier
    print('✅ Enhanced Sentiment Classifier imported')
except ImportError as e:
    print(f'❌ Enhanced Sentiment Classifier failed: {e}')

try:
    from src.models.enhanced_aspect_classifier import EnhancedAspectClassifier  
    print('✅ Enhanced Aspect Classifier imported')
except ImportError as e:
    print(f'❌ Enhanced Aspect Classifier failed: {e}')

try:
    from src.pipelines.integrated_ml_pipeline import IntegratedMLPipeline
    print('✅ Integrated ML Pipeline imported')
except ImportError as e:
    print(f'❌ Integrated ML Pipeline failed: {e}')

print('🎯 Import test complete!')
"