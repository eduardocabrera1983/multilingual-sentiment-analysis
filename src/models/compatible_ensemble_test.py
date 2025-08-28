#!/usr/bin/env python3
"""
Most Compatible Ensemble: Both models from CardiffNLP
This should eliminate output format inconsistencies
"""

import torch
from transformers import pipeline
import time

class CompatibleEnsemble:
    def __init__(self):
        self.models = {}
        self.ensemble_weights = {
            'xlm_roberta': 0.533,  # 40% / (40% + 35%) = 0.533
            'twitter_roberta': 0.467  # 35% / (40% + 35%) = 0.467
        }
    
    def load_models(self):
        """Load both CardiffNLP models for maximum compatibility"""
        print("Loading Compatible CardiffNLP Ensemble...")
        print("=" * 50)
        
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"Device: {device_name}")
        
        # Both models from same organization for consistency
        model_configs = {
            'xlm_roberta': {
                'model_id': 'cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual',
                'description': 'XLM-RoBERTa (Multilingual)',
                'weight': 0.533
            },
            'twitter_roberta': {
                'model_id': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'description': 'Twitter-RoBERTa (Latest)',
                'weight': 0.467
            }
        }
        
        total_load_time = 0
        
        for model_name, config in model_configs.items():
            print(f"\nLoading {config['description']} ({config['weight']*100:.1f}%)...")
            
            try:
                start_time = time.time()
                
                # Use consistent configuration for both models
                model = pipeline(
                    'sentiment-analysis',
                    model=config['model_id'],
                    device=device,
                    top_k=None,  # Get all scores
                    truncation=True,
                    max_length=512
                )
                
                load_time = time.time() - start_time
                total_load_time += load_time
                
                self.models[model_name] = {
                    'pipeline': model,
                    'weight': config['weight'],
                    'description': config['description'],
                    'model_id': config['model_id']
                }
                
                print(f"Loaded {config['description']} in {load_time:.1f}s")
                
                # Test output format consistency
                test_result = model("This is good")
                print(f"Output format: {test_result}")
                
            except Exception as e:
                print(f"Failed to load {config['description']}: {e}")
                return False
        
        print(f"\nTotal ensemble load time: {total_load_time:.1f}s")
        print(f"Weights: XLM-RoBERTa {self.ensemble_weights['xlm_roberta']*100:.1f}%, Twitter-RoBERTa {self.ensemble_weights['twitter_roberta']*100:.1f}%")
        
        return True
    
    def _extract_scores(self, result, model_name):
        """Extract sentiment scores with CardiffNLP format handling"""
        scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        try:
            # Handle nested lists
            items = result
            if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
                items = result[0]
            elif isinstance(result, list) and len(result) == 1:
                items = result
            
            print(f"  {model_name} processing {len(items)} items:")
            
            for item in items:
                if isinstance(item, dict):
                    label = item.get('label', '').lower()
                    score = float(item.get('score', 0.0))
                    
                    print(f"    {label}: {score:.4f}")
                    
                    # CardiffNLP models use consistent labels
                    if 'positive' in label:
                        scores['positive'] = score
                    elif 'negative' in label:
                        scores['negative'] = score
                    elif 'neutral' in label:
                        scores['neutral'] = score
            
            # Verify scores are reasonable
            total = sum(scores.values())
            if abs(total - 1.0) > 0.01:  # Allow small floating point errors
                print(f"  WARNING: {model_name} scores sum to {total:.3f}, not 1.0")
            
            print(f"  {model_name} final scores: {scores}")
            return scores
            
        except Exception as e:
            print(f"  Error extracting scores from {model_name}: {e}")
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
    
    def predict_ensemble(self, text):
        """Predict using compatible ensemble"""
        predictions = {}
        
        print(f"\nProcessing: '{text}'")
        print("-" * 50)
        
        # Get predictions from both models
        for model_name, model_info in self.models.items():
            try:
                print(f"Running {model_name}...")
                raw_result = model_info['pipeline'](text)
                
                scores = self._extract_scores(raw_result, model_name)
                
                predictions[model_name] = {
                    'scores': scores,
                    'weight': model_info['weight']
                }
                
            except Exception as e:
                print(f"Model {model_name} failed: {e}")
                continue
        
        if not predictions:
            return None
        
        return self._combine_predictions(predictions, text)
    
    def _combine_predictions(self, predictions, text):
        """Combine predictions with weighted averaging"""
        
        ensemble_scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        total_weight = 0.0
        
        print("\nCombining predictions:")
        
        for model_name, pred_info in predictions.items():
            scores = pred_info['scores']
            weight = pred_info['weight']
            
            individual_prediction = max(scores, key=scores.get)
            individual_confidence = scores[individual_prediction]
            
            print(f"  {model_name}: {individual_prediction} ({individual_confidence:.3f}) [weight: {weight:.3f}]")
            
            # Add weighted contribution
            for sentiment in ensemble_scores:
                ensemble_scores[sentiment] += scores[sentiment] * weight
            
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            for sentiment in ensemble_scores:
                ensemble_scores[sentiment] /= total_weight
        
        # Final decision
        final_sentiment = max(ensemble_scores, key=ensemble_scores.get)
        final_confidence = ensemble_scores[final_sentiment]
        
        print(f"\nEnsemble result:")
        print(f"  Scores: {ensemble_scores}")
        print(f"  Prediction: {final_sentiment} ({final_confidence:.3f})")
        
        return {
            'ensemble_prediction': final_sentiment,
            'ensemble_confidence': final_confidence,
            'ensemble_scores': ensemble_scores,
            'individual_results': {name: info['scores'] for name, info in predictions.items()},
            'weights': {name: info['weight'] for name, info in predictions.items()}
        }

def test_compatible_ensemble():
    """Test the compatible CardiffNLP ensemble"""
    
    ensemble = CompatibleEnsemble()
    
    if not ensemble.load_models():
        print("Failed to load compatible ensemble")
        return False
    
    # Test cases covering different sentiment scenarios
    test_texts = [
        "This app is absolutely terrible and keeps crashing!",
        "I love this app, it works perfectly and is very reliable!",
        "The app is okay, has some good features but also some issues",
        "not receiving email for sign in, this app continues to be trash!",
        "Great tracking but the interface design could be improved"
    ]
    
    print("\nTesting Compatible Ensemble:")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}")
        print('='*60)
        
        start_time = time.time()
        result = ensemble.predict_ensemble(text)
        pred_time = time.time() - start_time
        
        if result:
            print(f"\nFINAL RESULT:")
            print(f"  Text: {text}")
            print(f"  Prediction: {result['ensemble_prediction']}")
            print(f"  Confidence: {result['ensemble_confidence']:.3f}")
            print(f"  Processing time: {pred_time*1000:.1f}ms")
            
            # Show breakdown
            print(f"\nModel breakdown:")
            for model_name, scores in result['individual_results'].items():
                weight = result['weights'][model_name]
                pred = max(scores, key=scores.get)
                conf = scores[pred]
                print(f"  {model_name}: {pred} ({conf:.3f}) [weight: {weight:.3f}]")
                
        else:
            print("Prediction failed")
    
    return True

if __name__ == "__main__":
    print("Testing Most Compatible Ensemble: CardiffNLP Models")
    print("=" * 60)
    
    success = test_compatible_ensemble()
    
    if success:
        print("\n" + "=" * 60)
        print("Compatible ensemble test completed")
        print("Both models from CardiffNLP should have consistent output formats")
        print("Ready to integrate into your enhanced_sentiment_classifier.py")
    else:
        print("Compatibility test failed")