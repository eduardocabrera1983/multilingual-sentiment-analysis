from integrated_classifier import IntegratedReviewAnalyzer

# Initialize
analyzer = IntegratedReviewAnalyzer(use_gpu=True)

# Analyze single review
result = analyzer.analyze_single_review(
    "The app works so good I want to recommend it to all my colleagues."
)

# Check the results
print(f"Sentiment: {result['sentiment']['label']}")  # positive ✅
print(f"Confidence: {result['sentiment']['confidence_percentage']}")  # ~95% ✅
print(f"Classification: {result['aspects']['classification_type']}")  # dual_strengths ✅
print(f"Aspect Confidence: {result['aspects']['confidence_percentage']}")  # <100% ✅