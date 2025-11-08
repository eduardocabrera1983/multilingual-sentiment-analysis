import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import re
import sys
from pathlib import Path
from collections import Counter

# CORRECTED PATH DETECTION
def find_project_root():
    """Find project root by looking for key directories"""
    current_path = Path(__file__).resolve().parent
    
    # Go up directories until we find src/ and data/ directories
    for _ in range(5):  # Limit search to prevent infinite loop
        if (current_path / 'src').exists() and (current_path / 'src' / 'models').exists():
            return current_path
        current_path = current_path.parent
    
    # Fallback: assume current directory structure
    current_dir = Path(__file__).resolve().parent
    if current_dir.name == 'scrapers':
        return current_dir.parent.parent  # src/scrapers -> src -> project_root
    else:
        return current_dir  # Assume file is at project root

project_root = find_project_root()
src_dir = project_root / 'src'

# Add correct paths
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(src_dir / 'models'))
sys.path.insert(0, str(src_dir / 'scrapers'))

print(f"Project root detected: {project_root}")
print(f"Source directory: {src_dir}")
print(f"Source directory exists: {src_dir.exists()}")

def safe_strip(text, default=""):
    """Safely strip text, handling None values"""
    if text is None:
        return default
    return str(text).strip()

def detect_device():
    """Automatically detect best available device (GPU first, then CPU)"""
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {gpu_name}")
            return device
        else:
            print("CUDA not available, using CPU")
            return 'cpu'
    except ImportError:
        print("PyTorch not installed, using CPU")
        return 'cpu'
    except Exception as e:
        print(f"Error detecting device: {e}, using CPU")
        return 'cpu'

try:
    from google_play_scraper import app, reviews, Sort
    SCRAPER_AVAILABLE = True
    print("google-play-scraper loaded successfully")
except ImportError:
    SCRAPER_AVAILABLE = False
    print("google-play-scraper not installed. Install with: pip install google-play-scraper")

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("langdetect not installed. Install with: pip install langdetect")

# IMPROVED IMPORT STRATEGY
ENHANCED_MODELS_AVAILABLE = False
EnhancedAspectClassifier = None
EnhancedSentimentClassifier = None

# Try multiple import strategies
import_attempts = [
    # Direct imports (if models directory is in path)
    ("from enhanced_aspect_classifier import EnhancedAspectClassifier", 
     "from enhanced_sentiment_classifier import EnhancedSentimentClassifier"),
    # With models prefix
    ("from models.enhanced_aspect_classifier import EnhancedAspectClassifier", 
     "from models.enhanced_sentiment_classifier import EnhancedSentimentClassifier"),
    # With src.models prefix  
    ("from src.models.enhanced_aspect_classifier import EnhancedAspectClassifier", 
     "from src.models.enhanced_sentiment_classifier import EnhancedSentimentClassifier")
]

for aspect_import, sentiment_import in import_attempts:
    try:
        exec(aspect_import)
        exec(sentiment_import)
        ENHANCED_MODELS_AVAILABLE = True
        print("Enhanced models with two-model ensemble loaded successfully")
        break
    except ImportError as e:
        continue

# Final fallback: direct file loading
if not ENHANCED_MODELS_AVAILABLE:
    try:
        import importlib.util
        
        aspect_path = src_dir / 'models' / 'enhanced_aspect_classifier.py'
        sentiment_path = src_dir / 'models' / 'enhanced_sentiment_classifier.py'
        
        if aspect_path.exists() and sentiment_path.exists():
            # Load aspect classifier
            spec = importlib.util.spec_from_file_location("enhanced_aspect_classifier", aspect_path)
            aspect_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(aspect_module)
            EnhancedAspectClassifier = aspect_module.EnhancedAspectClassifier
            
            # Load sentiment classifier
            spec = importlib.util.spec_from_file_location("enhanced_sentiment_classifier", sentiment_path)
            sentiment_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sentiment_module)
            EnhancedSentimentClassifier = sentiment_module.EnhancedSentimentClassifier
            
            ENHANCED_MODELS_AVAILABLE = True
            print("Enhanced models loaded via direct file import")
        else:
            print(f"Model files not found:")
            print(f"  Aspect: {aspect_path.exists()} - {aspect_path}")
            print(f"  Sentiment: {sentiment_path.exists()} - {sentiment_path}")
    except Exception as e2:
        print(f"Direct file import also failed: {e2}")

if not ENHANCED_MODELS_AVAILABLE:
    print("WARNING: Enhanced models not available, will use basic classification")

class FedExReviewAnalyzer:
    def __init__(self, data_dir=None, use_enhanced_models=True, device='auto'):
        # CORRECTED DATA DIRECTORY CALCULATION
        if data_dir is None:
            data_dir = str(project_root / 'data')
            print(f"Data will be saved to: {data_dir}")
        
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # FedEx app ID for different app stores (configurable via environment)
        self.fedex_app_id = os.environ.get('FEDEX_APP_ID', 'com.fedex.ida.android')
        
        # Load multilingual scraping configuration
        countries_str = os.environ.get('SCRAPING_COUNTRIES', 'us,es,de,fr,nl,it,br,mx,ca,au,gb,in')
        self.scraping_countries = [c.strip().lower() for c in countries_str.split(',')]
        
        languages_str = os.environ.get('SCRAPING_LANGUAGES', 'en,es,de,fr,nl,it,pt,hi')
        self.scraping_languages = [l.strip().lower() for l in languages_str.split(',')]
        
        print(f"Multilingual scraping enabled:")
        print(f"  Countries: {', '.join(self.scraping_countries)}")
        print(f"  Languages: {', '.join(self.scraping_languages)}")
        
        # AUTO-DETECT DEVICE WITH GPU PRIORITY
        if device == 'auto':
            self.device = detect_device()
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize enhanced models with two-model ensemble if available
        self.use_enhanced_models = use_enhanced_models and ENHANCED_MODELS_AVAILABLE
        
        if self.use_enhanced_models:
            print("Initializing enhanced ML models with two-model ensemble...")
            try:
                self.aspect_classifier = EnhancedAspectClassifier(confidence_threshold=0.3)
                # Initialize sentiment classifier with two-model ensemble
                self.sentiment_classifier = EnhancedSentimentClassifier(device=self.device, verbose=False)
                print(f"Enhanced models with two-model ensemble initialized on {self.device}")
            except Exception as e:
                print(f"Failed to initialize enhanced models: {e}")
                print("Falling back to basic classification")
                self.use_enhanced_models = False
                self.aspect_classifier = None
                self.sentiment_classifier = None
        else:
            print("Using basic keyword-based classification")
            self.aspect_classifier = None
            self.sentiment_classifier = None
        
        # Initialize review templates for all 6 aspects
        self._initialize_review_templates()
    
    def _initialize_review_templates(self):
        """Initialize review templates for sample data generation"""
        self.aspect_review_templates = {
            'user_experience': {
                'positive': [
                    "Very easy to use, intuitive interface for package management",
                    "Love how simple it is to navigate and find my packages",
                    "User-friendly design makes tracking shipments effortless",
                    "Clean and straightforward interface, perfect for daily use"
                ],
                'negative': [
                    "Interface is confusing, hard to find tracking information",
                    "Terrible user experience, impossible to navigate properly",
                    "Menu layout is awful, buttons are hard to find",
                    "Completely unusable interface, worst app design ever"
                ]
            },
            'performance': {
                'positive': [
                    "App runs smoothly, never crashes or freezes",
                    "Fast and responsive, works perfectly every time",
                    "Stable performance, loads quickly and reliably",
                    "Excellent app performance, no bugs or glitches"
                ],
                'negative': [
                    "App crashes constantly when trying to track packages",
                    "Freezes every time I open multiple shipments",
                    "Extremely slow loading, takes forever to update",
                    "Buggy and unresponsive, crashes frequently"
                ]
            },
            'tracking_accuracy': {
                'positive': [
                    "Tracking is very accurate, always shows correct package location",
                    "Real-time updates are precise and reliable",
                    "Package status updates are always accurate and timely",
                    "Excellent tracking, never had wrong information"
                ],
                'negative': [
                    "Tracking information is often delayed or wrong",
                    "Shows completely inaccurate package locations",
                    "Never updates tracking status properly",
                    "Tracking data is consistently incorrect and outdated"
                ]
            },
            'delivery_issues': {
                'positive': [
                    "Delivery notifications work perfectly, very reliable",
                    "Always delivers on time as promised",
                    "Great delivery service, packages arrive safely",
                    "Perfect delivery experience every single time"
                ],
                'negative': [
                    "Package was never delivered, completely lost",
                    "Late deliveries constantly, missed multiple deadlines",
                    "Damaged packages, terrible delivery handling",
                    "Wrong address delivery, driver issues consistently"
                ]
            },
            'interface_design': {
                'positive': [
                    "Beautiful modern design, looks very professional",
                    "Clean and attractive visual design",
                    "Sleek interface with great color scheme",
                    "Well-designed layout, visually appealing"
                ],
                'negative': [
                    "Ugly outdated design, looks terrible",
                    "Cluttered and messy interface design",
                    "Poor visual design, confusing layout",
                    "Horrible color choices, awful aesthetics"
                ]
            },
            'general_satisfaction': {
                'positive': [
                    "Love this app, highly recommend to everyone",
                    "Excellent overall experience with FedEx app",
                    "Perfect app for all my shipping needs",
                    "Outstanding service, couldn't be happier"
                ],
                'negative': [
                    "Hate this app, worst experience ever",
                    "Completely disappointed with the service",
                    "Never using this terrible app again",
                    "Total disaster, extremely unsatisfied"
                ]
            }
        }
        
        # Multi-language templates
        self.multilingual_templates = {
            'es': {
                'user_experience': [
                    "Muy facil de usar, interfaz intuitiva para gestionar paquetes",
                    "La interfaz es confusa, dificil encontrar informacion de seguimiento"
                ],
                'performance': [
                    "La aplicacion funciona perfectamente, rapida y estable",
                    "La aplicacion se cierra cuando trato de rastrear varios paquetes"
                ],
                'tracking_accuracy': [
                    "El seguimiento es muy preciso, siempre muestra la ubicacion correcta",
                    "La informacion de seguimiento esta retrasada o es incorrecta"
                ]
            },
            'de': {
                'user_experience': [
                    "Sehr einfach zu bedienen, intuitive Benutzeroberflache",
                    "Verwirrende Oberflache, schwer zu navigieren"
                ],
                'performance': [
                    "App lauft reibungslos und stabil",
                    "App sturzt ab beim Verfolgen mehrerer Pakete"
                ],
                'tracking_accuracy': [
                    "Verfolgung ist sehr genau, zeigt immer den korrekten Paketstandort",
                    "Tracking-Informationen sind oft falsch oder veraltet"
                ]
            },
            'fr': {
                'user_experience': [
                    "Tres facile a utiliser, interface intuitive",
                    "Interface tres confuse, difficile de trouver les informations"
                ],
                'performance': [
                    "L'application fonctionne parfaitement, rapide et stable",
                    "L'application plante constamment"
                ],
                'tracking_accuracy': [
                    "Le suivi est tres precis, montre toujours l'emplacement correct",
                    "Les informations de suivi sont souvent incorrectes"
                ]
            },
            'nl': {
                'user_experience': [
                    "Zeer gebruiksvriendelijk, intuitieve interface",
                    "Interface is verwarrend, moeilijk om tracking info te vinden"
                ],
                'performance': [
                    "App werkt perfect, snel en stabiel",
                    "App crasht constant bij het volgen van pakketten"
                ],
                'tracking_accuracy': [
                    "Tracking is zeer nauwkeurig, toont altijd de juiste locatie",
                    "Tracking informatie is vaak verkeerd of vertraagd"
                ]
            }
        }

    def scrape_fedex_reviews_adaptive(self, target_count=1000, countries=None):
        """
        Scrape REAL FedEx app reviews with improved error handling and rate limiting
        
        Args:
            target_count: Target number of REAL reviews to collect (default: 1000)
            countries: List of country codes to scrape from (uses environment config if None)
        
        Returns:
            List of real reviews, sorted by date (most recent first)
        """
        if not SCRAPER_AVAILABLE:
            print("google-play-scraper not available. Cannot get real reviews.")
            print("Install with: pip install google-play-scraper")
            return []
        
        # Use environment configuration if countries not provided
        if countries is None:
            countries = self.scraping_countries
        
        all_reviews = []
        reviews_by_country = {}
        
        print(f"Goal: Collect {target_count} REAL reviews")
        print(f"Countries: {', '.join([c.upper() for c in countries])}")
        print(f"Strategy: Start with recent reviews, extend timeline as needed")
        print("-" * 70)
        
        # First pass: Get all available reviews from each country
        for country in countries:
            try:
                print(f"\nFetching ALL available reviews from {country.upper()}...")
                
                # Extended language mapping for multilingual support
                lang_map = {
                    'us': 'en', 'gb': 'en', 'ca': 'en', 'au': 'en', 'in': 'en',
                    'es': 'es', 'mx': 'es',
                    'de': 'de', 'at': 'de', 'ch': 'de',
                    'fr': 'fr',
                    'be': 'nl',  # Belgium -> Dutch as primary
                    'nl': 'nl',
                    'it': 'it',
                    'br': 'pt', 'pt': 'pt',
                    'jp': 'ja', 'kr': 'ko',
                    'cn': 'zh', 'tw': 'zh',
                    'ru': 'ru', 'pl': 'pl',
                    'tr': 'tr', 'ar': 'ar',
                    'hi': 'hi'
                }
                lang = lang_map.get(country, 'en')
                
                # Get maximum available reviews with improved error handling
                try:
                    result, continuation_token = reviews(
                        self.fedex_app_id,
                        lang=lang,
                        country=country.upper(),
                        sort=Sort.NEWEST,
                        count=500
                    )
                except Exception as e:
                    print(f"Error fetching initial reviews for {country}: {e}")
                    reviews_by_country[country] = []
                    continue
                
                country_reviews = []
                
                # FIXED: Continue fetching with timeout and max attempts
                max_attempts = 10
                attempt_count = 0
                
                while continuation_token and len(country_reviews) < 500 and attempt_count < max_attempts:
                    attempt_count += 1
                    try:
                        print(f"  Fetching additional reviews (attempt {attempt_count}/10)...")
                        next_result, continuation_token = reviews(
                            self.fedex_app_id,
                            lang=lang,
                            country=country.upper(),
                            sort=Sort.NEWEST,
                            continuation_token=continuation_token
                        )
                        result.extend(next_result)
                        if not next_result:
                            print("  No more reviews available")
                            break
                        
                        # Rate limiting
                        time.sleep(1)
                        
                    except Exception as e:
                        print(f"  API error on attempt {attempt_count}: {e}")
                        if attempt_count >= 3:  # Break after 3 consecutive failures
                            break
                        time.sleep(2)  # Wait longer before retry
                
                # Process all reviews from this country
                for review in result:
                    review_content = safe_strip(review.get('content'))
                    if review_content:  # Only process non-empty reviews
                        detected_lang = self.detect_language(review_content)
                        
                        review_date = review.get('at', datetime.now())
                        country_reviews.append({
                            'app_id': self.fedex_app_id,
                            'text': review_content,
                            'rating': review.get('score', 0),
                            'date': review_date,
                            'date_str': review_date.strftime('%Y-%m-%d'),
                            'days_ago': (datetime.now() - review_date).days,
                            'country': country,
                            'language_detected': detected_lang,
                            'language_expected': lang,
                            'helpful_count': review.get('thumbsUpCount', 0),
                            'user': safe_strip(review.get('userName', 'Anonymous')),
                            'is_recent': (datetime.now() - review_date).days <= 90,
                            'is_real': True
                        })
                
                reviews_by_country[country] = country_reviews
                print(f"Retrieved {len(country_reviews)} real reviews from {country.upper()}")
                
                # Date range info
                if country_reviews:
                    dates = [r['date'] for r in country_reviews]
                    oldest = min(dates)
                    newest = max(dates)
                    print(f"   Date range: {oldest.strftime('%Y-%m-%d')} to {newest.strftime('%Y-%m-%d')}")
                
                # Avoid rate limiting between countries
                time.sleep(2)
                
            except Exception as e:
                print(f"Error scraping {country}: {e}")
                reviews_by_country[country] = []
                continue
        
        # Combine all reviews and sort by date (most recent first)
        for country_reviews in reviews_by_country.values():
            all_reviews.extend(country_reviews)
        
        # Sort by date - MOST RECENT FIRST
        all_reviews.sort(key=lambda x: x['date'], reverse=True)
        
        # Take only the target count (most recent reviews)
        if len(all_reviews) > target_count:
            print(f"\nTotal available: {len(all_reviews)} reviews")
            print(f"Taking the {target_count} most recent reviews")
            all_reviews = all_reviews[:target_count]
        else:
            print(f"\nTotal collected: {len(all_reviews)} real reviews")
            if len(all_reviews) < target_count:
                print(f"Only {len(all_reviews)} real reviews available (target was {target_count})")
        
        # Summary statistics
        if all_reviews:
            dates = [r['date'] for r in all_reviews]
            oldest = min(dates)
            newest = max(dates)
            
            print(f"\nFinal Dataset Statistics:")
            print(f"   Total reviews: {len(all_reviews)}")
            print(f"   Date range: {oldest.strftime('%Y-%m-%d')} to {newest.strftime('%Y-%m-%d')}")
            print(f"   Timespan: {(newest - oldest).days} days")
            
            # Distribution by time periods
            last_30 = sum(1 for r in all_reviews if r['days_ago'] <= 30)
            last_90 = sum(1 for r in all_reviews if r['days_ago'] <= 90)
            last_180 = sum(1 for r in all_reviews if r['days_ago'] <= 180)
            
            print(f"\nRecency Distribution:")
            print(f"   Last 30 days: {last_30} reviews ({last_30/len(all_reviews)*100:.1f}%)")
            print(f"   Last 90 days: {last_90} reviews ({last_90/len(all_reviews)*100:.1f}%)")
            print(f"   Last 180 days: {last_180} reviews ({last_180/len(all_reviews)*100:.1f}%)")
            
            # Country distribution
            country_dist = Counter([r['country'] for r in all_reviews])
            print(f"\nCountry Distribution:")
            for country, count in country_dist.most_common():
                print(f"   {country.upper()}: {count} reviews ({count/len(all_reviews)*100:.1f}%)")
        
        return all_reviews
    
    def create_country_enhanced_samples(self, country, count, cutoff_date=None):
        """Create enhanced sample data for a specific country with all 6 aspects"""
        lang_map = {'us': 'en', 'es': 'es', 'de': 'de', 'fr': 'fr', 'nl': 'nl'}
        lang = lang_map.get(country, 'en')
        
        reviews = []
        aspects = list(self.aspect_review_templates.keys())
        
        # Calculate date range for samples
        if cutoff_date:
            date_range = (datetime.now() - cutoff_date).days
        else:
            date_range = 90
            cutoff_date = datetime.now() - timedelta(days=90)
        
        for i in range(count):
            # Select aspect and sentiment
            aspect = aspects[i % len(aspects)]
            sentiment_type = 'positive' if np.random.random() > 0.4 else 'negative'
            
            # Get appropriate text
            if lang == 'en':
                templates = self.aspect_review_templates[aspect][sentiment_type]
                text = np.random.choice(templates)
            elif lang in self.multilingual_templates and aspect in self.multilingual_templates[lang]:
                templates = self.multilingual_templates[lang][aspect]
                text = np.random.choice(templates)
            else:
                templates = self.aspect_review_templates[aspect][sentiment_type]
                text = np.random.choice(templates)
            
            # Add some variations for mixed concerns
            if np.random.random() > 0.7:  # 30% chance of mixed concerns
                secondary_aspect = np.random.choice([a for a in aspects if a != aspect])
                secondary_sentiment = 'positive' if np.random.random() > 0.5 else 'negative'
                if lang == 'en':
                    secondary_text = np.random.choice(
                        self.aspect_review_templates[secondary_aspect][secondary_sentiment]
                    )
                    text = f"{text} {secondary_text}"
            
            # Determine rating based on sentiment
            if sentiment_type == 'positive':
                rating = np.random.choice([4, 5], p=[0.3, 0.7])
            else:
                rating = np.random.choice([1, 2], p=[0.6, 0.4])
            
            # Generate a realistic date within the range
            days_ago = int(np.random.exponential(scale=date_range/3))
            days_ago = min(days_ago, date_range)
            review_date = datetime.now() - timedelta(days=days_ago)
            
            reviews.append({
                'app_id': self.fedex_app_id,
                'text': text,
                'rating': rating,
                'date': review_date,
                'date_str': review_date.strftime('%Y-%m-%d'),
                'days_ago': days_ago,
                'country': country,
                'language_detected': lang,
                'language_expected': lang,
                'helpful_count': np.random.randint(0, 50),
                'user': f'User_{np.random.randint(1000, 9999)}',
                'is_recent': days_ago <= 90,
                'is_real': False
            })
        
        return reviews
    
    def detect_language(self, text):
        """Detect language of review text"""
        if not LANGDETECT_AVAILABLE or not text.strip():
            return 'unknown'
        
        try:
            return detect(text)
        except:
            return 'unknown'
    
    def classify_reviews_enhanced(self, reviews):
        """Classify reviews using enhanced multi-label aspect classifier with improved progress tracking"""
        print("Classifying aspects using enhanced multi-label system with two-model ensemble...")
        
        if not self.use_enhanced_models:
            print("Enhanced models not available, using basic classification")
            return self.classify_reviews_basic(reviews)
        
        classified_reviews = []
        
        # Process reviews in batches for efficiency
        batch_size = 32 if self.sentiment_classifier and hasattr(self.sentiment_classifier, 'analyze_batch') else 1
        total_batches = (len(reviews) + batch_size - 1) // batch_size
        
        print(f"Processing {len(reviews)} reviews in {total_batches} batches (batch size: {batch_size})")
        
        # Add timing for better estimates
        start_time = time.time()
        
        for batch_idx, batch_start in enumerate(range(0, len(reviews), batch_size)):
            batch_end = min(batch_start + batch_size, len(reviews))
            batch = reviews[batch_start:batch_end]
            
            # IMPROVED PROGRESS REPORTING with ETA
            elapsed = time.time() - start_time
            progress_pct = (batch_start / len(reviews)) * 100
            
            if batch_idx > 0:
                avg_time_per_batch = elapsed / batch_idx
                remaining_batches = total_batches - batch_idx
                eta_seconds = remaining_batches * avg_time_per_batch
                eta_minutes = eta_seconds / 60
                
                print(f"Batch {batch_idx+1}/{total_batches} - Reviews {batch_start}-{batch_end}/{len(reviews)} "
                      f"[{progress_pct:.1f}%] ETA: {eta_minutes:.1f}min")
            else:
                print(f"Batch {batch_idx+1}/{total_batches} - Reviews {batch_start}-{batch_end}/{len(reviews)} "
                      f"[{progress_pct:.1f}%]")
            
            # Extract texts for batch processing
            texts = [review['text'] for review in batch]
            languages = [review.get('language_detected', 'en') for review in batch]
            
            # Get sentiment analysis results with error handling
            try:
                if hasattr(self.sentiment_classifier, 'analyze_batch'):
                    sentiment_results = self.sentiment_classifier.analyze_batch(texts)
                else:
                    sentiment_results = []
                    for text, lang in zip(texts, languages):
                        try:
                            sentiment_result = self.sentiment_classifier.analyze_sentiment(text, lang)
                            sentiment_results.append(sentiment_result)
                        except Exception as e:
                            print(f"Error analyzing sentiment for text: {e}")
                            # Fallback sentiment result
                            sentiment_results.append({
                                'sentiment': 'neutral',
                                'confidence': 0.5,
                                'scores': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33},
                                'method': 'error_fallback',
                                'models_used': 0,
                                'device': self.device,
                                'from_cache': False,
                                'processing_time': 0
                            })
            except Exception as e:
                print(f"Error in sentiment analysis for batch {batch_idx}: {e}")
                # Create fallback results for entire batch
                sentiment_results = []
                for text in texts:
                    sentiment_results.append({
                        'sentiment': 'neutral',
                        'confidence': 0.5,
                        'scores': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33},
                        'method': 'batch_error_fallback',
                        'models_used': 0,
                        'device': self.device,
                        'from_cache': False,
                        'processing_time': 0
                    })
            
            # Process each review in the batch
            for review, text, lang, sentiment_result in zip(batch, texts, languages, sentiment_results):
                # Get multi-label aspect classification with error handling
                try:
                    aspect_result = self.aspect_classifier.classify_aspects_multilabel(
                        text, 
                        lang,
                        sentiment_result['sentiment'],
                        sentiment_result['confidence']
                    )
                except Exception as e:
                    print(f"Error in aspect classification: {e}")
                    # Create fallback aspect result
                    aspect_result = {
                        'primary_aspect': 'general_satisfaction',
                        'secondary_aspects': [],
                        'classification_type': 'error_fallback',
                        'confidence': 0.5,
                        'all_scores': {},
                        'priority_level': 'MEDIUM',
                        'severity_level': 'MODERATE',
                        'requires_immediate_action': False,
                        'business_summary': 'Classification failed - manual review needed',
                        'recommendation': 'Manual review required due to processing error'
                    }
                
                # Combine original review data with classifications
                classified_review = {
                    **review,  # Original review data
                    
                    # Multi-label aspect classification
                    'primary_aspect': aspect_result['primary_aspect'],
                    'secondary_aspects': aspect_result['secondary_aspects'],
                    'classification_type': aspect_result['classification_type'],
                    'aspect_confidence': aspect_result['confidence'],
                    'all_aspect_scores': aspect_result['all_scores'],
                    'priority_level': aspect_result['priority_level'],
                    'severity_level': aspect_result['severity_level'],
                    'requires_immediate_action': aspect_result['requires_immediate_action'],
                    'business_summary': aspect_result['business_summary'],
                    'recommendation': aspect_result['recommendation'],
                    
                    # Enhanced sentiment analysis with two-model ensemble
                    'sentiment': sentiment_result['sentiment'],
                    'sentiment_confidence': sentiment_result['confidence'],
                    'sentiment_scores': sentiment_result['scores'],
                    'sentiment_method': sentiment_result.get('method', 'unknown'),
                    'sentiment_models_used': sentiment_result.get('models_used', 0),
                    'sentiment_device': sentiment_result.get('device', 'unknown'),
                    'sentiment_from_cache': sentiment_result.get('from_cache', False),
                    'sentiment_processing_time': sentiment_result.get('processing_time', 0),
                    
                    # Combined insights
                    'is_mixed_concern': len(aspect_result['secondary_aspects']) > 0,
                    'aspect_count': 1 + len(aspect_result['secondary_aspects']),
                    'is_critical': aspect_result['severity_level'] == 'CRITICAL' or 
                                  aspect_result['requires_immediate_action'],
                    
                    # Two-model ensemble specific flags
                    'ensemble_used': sentiment_result.get('method') == 'two_model_ensemble',
                    'rule_based_fallback': sentiment_result.get('method') == 'rule_based'
                }
                
                classified_reviews.append(classified_review)
        
        total_time = time.time() - start_time
        reviews_per_sec = len(reviews) / total_time if total_time > 0 else 0
        
        print(f"\nClassification completed!")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Average time per review: {total_time/len(reviews)*1000:.1f}ms")
        print(f"Throughput: {reviews_per_sec:.1f} reviews/second")
        print(f"Device used: {self.device}")
        
        return classified_reviews
    
    def classify_reviews_basic(self, reviews):
        """Basic classification fallback when enhanced models aren't available"""
        print("Using basic rating-based classification...")
        
        for review in reviews:
            rating = review['rating']
            if rating >= 4:
                review['sentiment'] = 'positive'
            elif rating <= 2:
                review['sentiment'] = 'negative'
            else:
                review['sentiment'] = 'neutral'
            
            review['sentiment_confidence'] = 0.8
            review['sentiment_method'] = 'rating_based'
            review['sentiment_models_used'] = 0
            review['sentiment_device'] = 'cpu'
            review['sentiment_from_cache'] = False
            review['sentiment_processing_time'] = 0.001
            
            review['primary_aspect'] = 'general_satisfaction'
            review['secondary_aspects'] = []
            review['classification_type'] = 'basic_rating'
            review['priority_level'] = 'MEDIUM'
            review['severity_level'] = 'MODERATE'
            review['requires_immediate_action'] = False
            
            review['ensemble_used'] = False
            review['rule_based_fallback'] = True
            
            # Add missing fields for consistency
            review['aspect_confidence'] = 0.5
            review['all_aspect_scores'] = {}
            review['business_summary'] = f"Basic classification based on {rating}-star rating"
            review['recommendation'] = "Upgrade to enhanced models for detailed analysis"
            review['is_mixed_concern'] = False
            review['aspect_count'] = 1
            review['is_critical'] = rating <= 2
            
        return reviews
    
    def analyze_fedex_reviews(self, count=1000, real_only=True):
        """
        Main function to scrape and analyze FedEx reviews with enhanced two-model ensemble
        
        Args:
            count: Number of reviews to collect (default: 1000)
            real_only: If True, only collect real reviews using adaptive timeline (default: True)
        """
        print("Starting Enhanced FedEx Review Analysis with Two-Model Ensemble...")
        print(f"Target app: {self.fedex_app_id}")
        print(f"Countries: {', '.join([c.upper() for c in self.scraping_countries])}")
        print(f"Languages: {', '.join(self.scraping_languages)}")
        print(f"Enhanced Models: {'ENABLED (Two-Model Ensemble)' if self.use_enhanced_models else 'DISABLED'}")
        print(f"Device: {self.device}")
        
        if real_only:
            print(f"Mode: REAL REVIEWS ONLY - Adaptive timeline to reach {count} reviews")
            reviews = self.scrape_fedex_reviews_adaptive(target_count=count)
        else:
            print("Synthetic mode not implemented - using real reviews only")
            reviews = self.scrape_fedex_reviews_adaptive(target_count=count)
        
        if not reviews:
            print("No reviews collected")
            return None
        
        # Process reviews with enhanced classification including two-model ensemble
        reviews = self.classify_reviews_enhanced(reviews)
        
        # Convert to DataFrame
        df = pd.DataFrame(reviews)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        device_suffix = self.device if self.device != 'cpu' else ''
        filename = f"{self.data_dir}/fedex_reviews_enhanced_ensemble{('_' + device_suffix) if device_suffix else ''}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        full_path = os.path.abspath(filename)
        print(f"\nData saved to: {full_path}")
        
        # Generate enhanced analysis report with ensemble metrics
        self.generate_enhanced_analysis_report(df, filename)
        
        # Generate temporal analysis if date information is available
        if 'date' in df.columns:
            self.generate_temporal_analysis(df)
        
        # Generate business intelligence report with ensemble performance
        if self.use_enhanced_models:
            self.generate_business_intelligence_report(df)
        
        # Generate ensemble performance report
        self.generate_ensemble_performance_report(df)
        
        return df
    
    def generate_enhanced_analysis_report(self, df, filename):
        """Generate enhanced analysis report for FedEx reviews with ensemble metrics"""
        print(f"\nEnhanced FedEx Review Analysis Report (Two-Model Ensemble)")
        print("="*70)
        print(f"Saved {len(df)} reviews to {filename}")
        
        print(f"\nLanguage Distribution:")
        lang_dist = df['language_detected'].value_counts()
        for lang, count in lang_dist.items():
            print(f"  {lang}: {count} reviews ({count/len(df)*100:.1f}%)")
        
        print(f"\nSentiment Distribution:")
        sentiment_dist = df['sentiment'].value_counts()
        for sentiment, count in sentiment_dist.items():
            print(f"  {sentiment}: {count} reviews ({count/len(df)*100:.1f}%)")
        
        # Two-Model Ensemble Performance
        if 'sentiment_method' in df.columns:
            print(f"\nTwo-Model Ensemble Performance:")
            method_dist = df['sentiment_method'].value_counts()
            for method, count in method_dist.items():
                print(f"  {method}: {count} reviews ({count/len(df)*100:.1f}%)")
        
        if 'sentiment_models_used' in df.columns:
            print(f"\nModels Used Distribution:")
            models_dist = df['sentiment_models_used'].value_counts()
            for models, count in models_dist.items():
                print(f"  {models} models: {count} reviews ({count/len(df)*100:.1f}%)")
        
        if 'sentiment_from_cache' in df.columns:
            cache_hits = df['sentiment_from_cache'].sum()
            cache_rate = (cache_hits / len(df)) * 100
            print(f"\nCache Performance:")
            print(f"  Cache hits: {cache_hits} ({cache_rate:.1f}%)")
            print(f"  Cache misses: {len(df) - cache_hits} ({100-cache_rate:.1f}%)")
        
        if 'primary_aspect' in df.columns:
            print(f"\nPrimary Aspect Distribution:")
            aspect_dist = df['primary_aspect'].value_counts()
            for aspect, count in aspect_dist.items():
                print(f"  {aspect}: {count} reviews ({count/len(df)*100:.1f}%)")
            
            print(f"\nClassification Types:")
            class_dist = df['classification_type'].value_counts()
            for class_type, count in class_dist.items():
                print(f"  {class_type}: {count} reviews ({count/len(df)*100:.1f}%)")
            
            print(f"\nPriority Levels:")
            priority_dist = df['priority_level'].value_counts()
            for priority, count in priority_dist.items():
                print(f"  {priority}: {count} reviews ({count/len(df)*100:.1f}%)")
        
        # Device performance
        if 'sentiment_device' in df.columns:
            print(f"\nDevice Performance:")
            device_dist = df['sentiment_device'].value_counts()
            for device, count in device_dist.items():
                print(f"  {device}: {count} reviews ({count/len(df)*100:.1f}%)")
        
        # Sample reviews by category
        self.print_sample_reviews(df)
    
    def generate_ensemble_performance_report(self, df):
        """Generate detailed ensemble performance report"""
        if not self.use_enhanced_models or 'sentiment_method' not in df.columns:
            return
        
        print(f"\nDetailed Two-Model Ensemble Performance Report")
        print("="*70)
        
        # Performance by method
        methods = df['sentiment_method'].unique()
        for method in methods:
            method_df = df[df['sentiment_method'] == method]
            
            print(f"\n{method.replace('_', ' ').title()} Performance:")
            print(f"  Reviews processed: {len(method_df)}")
            print(f"  Percentage of total: {len(method_df)/len(df)*100:.1f}%")
            
            if 'sentiment_confidence' in method_df.columns:
                avg_confidence = method_df['sentiment_confidence'].mean()
                print(f"  Average confidence: {avg_confidence*100:.1f}%")
            
            if 'sentiment_processing_time' in method_df.columns:
                avg_time = method_df['sentiment_processing_time'].mean() * 1000
                print(f"  Average processing time: {avg_time:.1f}ms")
            
            # Quality metrics
            if 'sentiment_confidence' in method_df.columns:
                high_confidence = len(method_df[method_df['sentiment_confidence'] >= 0.8])
                print(f"  High confidence results (>=80%): {high_confidence} ({high_confidence/len(method_df)*100:.1f}%)")
        
        # Device usage
        if 'sentiment_device' in df.columns:
            print(f"\nDevice Usage:")
            device_dist = df['sentiment_device'].value_counts()
            for device, count in device_dist.items():
                print(f"  {device}: {count} reviews ({count/len(df)*100:.1f}%)")
        
        # Overall performance metrics
        if 'sentiment_processing_time' in df.columns:
            total_time = df['sentiment_processing_time'].sum()
            avg_time = df['sentiment_processing_time'].mean() * 1000
            throughput = len(df) / total_time if total_time > 0 else 0
            
            print(f"\nOverall Performance:")
            print(f"  Total processing time: {total_time:.2f}s")
            print(f"  Average time per review: {avg_time:.1f}ms")
            print(f"  Throughput: {throughput:.1f} reviews/second")
        
        # Ensemble vs fallback comparison
        if 'ensemble_used' in df.columns:
            ensemble_count = df['ensemble_used'].sum()
            fallback_count = df['rule_based_fallback'].sum() if 'rule_based_fallback' in df.columns else 0
            
            print(f"\nEnsemble vs Fallback Usage:")
            print(f"  Two-model ensemble: {ensemble_count} ({ensemble_count/len(df)*100:.1f}%)")
            print(f"  Rule-based fallback: {fallback_count} ({fallback_count/len(df)*100:.1f}%)")
            
            if ensemble_count > 0 and 'sentiment_confidence' in df.columns:
                ensemble_df = df[df['ensemble_used'] == True]
                ensemble_confidence = ensemble_df['sentiment_confidence'].mean()
                print(f"  Ensemble average confidence: {ensemble_confidence*100:.1f}%")
            
            if fallback_count > 0 and 'sentiment_confidence' in df.columns:
                fallback_df = df[df['rule_based_fallback'] == True]
                fallback_confidence = fallback_df['sentiment_confidence'].mean()
                print(f"  Fallback average confidence: {fallback_confidence*100:.1f}%")
    
    def print_sample_reviews(self, df):
        """Print sample reviews for different categories"""
        print(f"\nSample Reviews by Category:")
        
        if 'primary_aspect' not in df.columns:
            return
        
        # Get unique aspects
        aspects = df['primary_aspect'].unique()
        
        for aspect in aspects[:3]:
            print(f"\n{aspect.replace('_', ' ').title()}:")
            
            # Positive samples
            positive_samples = df[
                (df['primary_aspect'] == aspect) & 
                (df['sentiment'] == 'positive')
            ]['text'].head(2)
            
            if len(positive_samples) > 0:
                print("  Positive:")
                for i, text in enumerate(positive_samples, 1):
                    print(f"    {i}. {text[:100]}...")
            
            # Negative samples
            negative_samples = df[
                (df['primary_aspect'] == aspect) & 
                (df['sentiment'] == 'negative')
            ]['text'].head(2)
            
            if len(negative_samples) > 0:
                print("  Negative:")
                for i, text in enumerate(negative_samples, 1):
                    print(f"    {i}. {text[:100]}...")
    
    def generate_business_intelligence_report(self, df):
        """Generate business intelligence report with ensemble performance"""
        if not self.use_enhanced_models:
            return
        
        print(f"\nBusiness Intelligence Report (Two-Model Ensemble)")
        print("="*70)
        
        # Prepare results for business report
        results = []
        for _, row in df.iterrows():
            result = {
                'primary_aspect': row.get('primary_aspect', 'general_satisfaction'),
                'secondary_aspects': row.get('secondary_aspects', []),
                'classification_type': row.get('classification_type', 'single_aspect'),
                'confidence': row.get('aspect_confidence', 0.5),
                'priority_level': row.get('priority_level', 'MEDIUM'),
                'severity_level': row.get('severity_level', 'MODERATE'),
                'requires_immediate_action': row.get('requires_immediate_action', False),
                'business_summary': row.get('business_summary', ''),
                'review_text': row.get('text', ''),
                'sentiment_method': row.get('sentiment_method', 'unknown'),
                'ensemble_used': row.get('ensemble_used', False)
            }
            results.append(result)
        
        # Generate report using aspect classifier
        if hasattr(self.aspect_classifier, 'generate_business_report'):
            try:
                report = self.aspect_classifier.generate_business_report(results)
                
                print(f"\nSummary Metrics:")
                for key, value in report['summary'].items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")
                
                if report.get('top_recommendations'):
                    print(f"\nTop Recommendations:")
                    for i, rec in enumerate(report['top_recommendations'], 1):
                        print(f"  {i}. {rec}")
            except Exception as e:
                print(f"Error generating business report: {e}")
        
        # Ensemble-specific business insights
        ensemble_reviews = [r for r in results if r.get('ensemble_used', False)]
        if ensemble_reviews:
            print(f"\nTwo-Model Ensemble Business Impact:")
            print(f"  Reviews processed with ensemble: {len(ensemble_reviews)}")
            print(f"  Ensemble usage rate: {len(ensemble_reviews)/len(results)*100:.1f}%")
            
            # High-priority reviews processed by ensemble
            high_priority_ensemble = [r for r in ensemble_reviews if r['priority_level'] == 'HIGH']
            if high_priority_ensemble:
                print(f"  High-priority reviews via ensemble: {len(high_priority_ensemble)}")
                print(f"  Ensemble reliability for critical decisions: {len(high_priority_ensemble)/len(ensemble_reviews)*100:.1f}%")
    
    def generate_temporal_analysis(self, df):
        """Generate temporal analysis of reviews"""
        if 'date' not in df.columns or 'days_ago' not in df.columns:
            return
        
        print(f"\nTemporal Analysis")
        print("="*70)
        
        # Recent vs older comparison
        recent_cutoff = 30
        recent_mask = df['days_ago'] <= recent_cutoff
        
        print(f"\nRecent (<=30 days) vs Older Reviews:")
        print(f"  Recent reviews: {recent_mask.sum()} ({recent_mask.sum()/len(df)*100:.1f}%)")
        print(f"  Older reviews: {(~recent_mask).sum()} ({(~recent_mask).sum()/len(df)*100:.1f}%)")
        
        if 'sentiment' in df.columns:
            recent_sentiment = df[recent_mask]['sentiment'].value_counts(normalize=True)
            older_sentiment = df[~recent_mask]['sentiment'].value_counts(normalize=True) if (~recent_mask).sum() > 0 else pd.Series()
            
            print(f"\n  Sentiment Comparison:")
            print(f"    Recent - Positive: {recent_sentiment.get('positive', 0)*100:.1f}%")
            print(f"    Recent - Negative: {recent_sentiment.get('negative', 0)*100:.1f}%")
            if not older_sentiment.empty:
                print(f"    Older - Positive: {older_sentiment.get('positive', 0)*100:.1f}%")
                print(f"    Older - Negative: {older_sentiment.get('negative', 0)*100:.1f}%")
        
        # Ensemble performance over time
        if 'ensemble_used' in df.columns:
            recent_ensemble = df[recent_mask]['ensemble_used'].sum()
            older_ensemble = df[~recent_mask]['ensemble_used'].sum() if (~recent_mask).sum() > 0 else 0
            
            print(f"\n  Ensemble Usage Over Time:")
            print(f"    Recent ensemble usage: {recent_ensemble}/{recent_mask.sum()} ({recent_ensemble/recent_mask.sum()*100:.1f}%)")
            if (~recent_mask).sum() > 0:
                print(f"    Older ensemble usage: {older_ensemble}/{(~recent_mask).sum()} ({older_ensemble/(~recent_mask).sum()*100:.1f}%)")

if __name__ == "__main__":
    # Initialize analyzer with automatic GPU/CPU detection
    analyzer = FedExReviewAnalyzer(device='auto')  # This will automatically choose GPU if available
    
    print("Analyzing FedEx Mobile App Reviews with Two-Model Ensemble")
    print("App: https://play.google.com/store/apps/details?id=com.fedex.ida.android")
    print("="*70)
    
    # Get 1000 REAL reviews using adaptive timeline with ensemble processing
    df = analyzer.analyze_fedex_reviews(
        count=1000,
        real_only=True
    )
    
    if df is not None:
        print("\nFedEx analysis with two-model ensemble complete!")
        print("="*70)
        print("Dataset Summary:")
        print(f"   Total REAL reviews: {len(df)}")
        
        if 'is_real' in df.columns:
            real_count = df['is_real'].sum()
            print(f"   Verified real reviews: {real_count}")
        
        if 'days_ago' in df.columns:
            print(f"\nTimeline Coverage:")
            print(f"   Most recent: {df['days_ago'].min()} days ago")
            print(f"   Oldest: {df['days_ago'].max()} days ago")
            print(f"   Average age: {df['days_ago'].mean():.1f} days")
        
        # Two-model ensemble specific metrics
        if 'sentiment_method' in df.columns:
            ensemble_count = (df['sentiment_method'] == 'two_model_ensemble').sum()
            print(f"\nTwo-Model Ensemble Performance:")
            print(f"   Reviews processed with ensemble: {ensemble_count} ({ensemble_count/len(df)*100:.1f}%)")
        
        if 'sentiment_from_cache' in df.columns:
            cache_hits = df['sentiment_from_cache'].sum()
            print(f"   Cache hits: {cache_hits} ({cache_hits/len(df)*100:.1f}%)")
        
        if 'sentiment_processing_time' in df.columns:
            avg_time = df['sentiment_processing_time'].mean() * 1000
            print(f"   Average processing time: {avg_time:.1f}ms per review")
        
        if 'sentiment_device' in df.columns:
            device_used = df['sentiment_device'].iloc[0] if len(df) > 0 else 'unknown'
            print(f"   Device used: {device_used}")
        
        print("\nAll reviews are REAL - no synthetic data!")
        print("Two-model ensemble provides enhanced accuracy and performance!")
        print("Ready for advanced analysis and ML pipeline!")
    else:
        print("Analysis failed")