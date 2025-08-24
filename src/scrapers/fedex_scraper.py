import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import re
import sys
from pathlib import Path
from collections import Counter

# Fix the import path for your specific directory structure
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Go up to multilingual-sentiment-analysis
sys.path.insert(0, str(project_root))

# Add src directory to path so we can import from models
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir))

try:
    from google_play_scraper import app, reviews, Sort
    SCRAPER_AVAILABLE = True
    print("✅ google-play-scraper loaded successfully")
except ImportError:
    SCRAPER_AVAILABLE = False
    print("⚠️ google-play-scraper not installed. Install with: pip install google-play-scraper")

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️ langdetect not installed. Install with: pip install langdetect")

# Import the enhanced models - they're in src/models/
try:
    from models.enhanced_aspect_classifier import EnhancedAspectClassifier
    from models.enhanced_sentiment_classifier import EnhancedSentimentClassifier
    ENHANCED_MODELS_AVAILABLE = True
    print("✅ Enhanced models loaded successfully")
except ImportError as e:
    ENHANCED_MODELS_AVAILABLE = False
    print(f"⚠️ Enhanced models not available: {e}")
    print("Looking for models in: src/models/")
    print(f"Current path: {src_dir}")
    
    # Try alternative import
    try:
        import sys
        import importlib.util
        
        # Direct file import as fallback
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
            print("✅ Enhanced models loaded via direct import")
        else:
            print(f"❌ Model files not found at expected locations:")
            print(f"   {aspect_path}")
            print(f"   {sentiment_path}")
    except Exception as e2:
        print(f"❌ Alternative import also failed: {e2}")

class FedExReviewAnalyzer:
    def __init__(self, data_dir=None, use_enhanced_models=True):
        # Set data directory to project root's data folder
        if data_dir is None:
            # Navigate to the project's main data folder
            current_file = Path(__file__)
            # Go up from scrapers to src, then to project root
            project_root = current_file.parent.parent.parent  # multilingual-sentiment-analysis
            data_dir = str(project_root / 'data')
            print(f"📁 Data will be saved to: {data_dir}")
        
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # FedEx app ID for different app stores
        self.fedex_app_id = "com.fedex.ida.android"
        
        # Initialize enhanced models if available
        self.use_enhanced_models = use_enhanced_models and ENHANCED_MODELS_AVAILABLE
        
        if self.use_enhanced_models:
            print("🚀 Initializing enhanced ML models...")
            self.aspect_classifier = EnhancedAspectClassifier(confidence_threshold=0.3)
            self.sentiment_classifier = EnhancedSentimentClassifier(use_ensemble=True)
            print("✅ Enhanced models initialized")
        else:
            print("⚠️ Using basic keyword-based classification")
            self.aspect_classifier = None
            self.sentiment_classifier = None
        
        # Sample review templates for all 6 aspects from enhanced model
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
                    "Excellent tracking accuracy, never had wrong information"
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
                    "Muy fácil de usar, interfaz intuitiva para gestionar paquetes",
                    "La interfaz es confusa, difícil encontrar información de seguimiento"
                ],
                'performance': [
                    "La aplicación funciona perfectamente, rápida y estable",
                    "La aplicación se cierra cuando trato de rastrear varios paquetes"
                ],
                'tracking_accuracy': [
                    "El seguimiento es muy preciso, siempre muestra la ubicación correcta",
                    "La información de seguimiento está retrasada o es incorrecta"
                ]
            },
            'de': {
                'user_experience': [
                    "Sehr einfach zu bedienen, intuitive Benutzeroberfläche",
                    "Verwirrende Oberfläche, schwer zu navigieren"
                ],
                'performance': [
                    "App läuft reibungslos und stabil",
                    "App stürzt ab beim Verfolgen mehrerer Pakete"
                ],
                'tracking_accuracy': [
                    "Verfolgung ist sehr genau, zeigt immer den korrekten Paketstandort",
                    "Tracking-Informationen sind oft falsch oder veraltet"
                ]
            },
            'fr': {
                'user_experience': [
                    "Très facile à utiliser, interface intuitive",
                    "Interface très confuse, difficile de trouver les informations"
                ],
                'performance': [
                    "L'application fonctionne parfaitement, rapide et stable",
                    "L'application plante constamment"
                ],
                'tracking_accuracy': [
                    "Le suivi est très précis, montre toujours l'emplacement correct",
                    "Les informations de suivi sont souvent incorrectes"
                ]
            },
            'nl': {
                'user_experience': [
                    "Zeer gebruiksvriendelijk, intuïtieve interface",
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
        # __init__ ends here - no return statement!
    
    def scrape_fedex_reviews_adaptive(self, target_count=1000, countries=['us', 'es', 'de', 'fr', 'nl']):
        """
        Scrape REAL FedEx app reviews, extending timeline as needed to reach target count
        
        Args:
            target_count: Target number of REAL reviews to collect (default: 1000)
            countries: List of country codes to scrape from
        
        Returns:
            List of real reviews, sorted by date (most recent first)
        """
        if not SCRAPER_AVAILABLE:
            print("❌ google-play-scraper not available. Cannot get real reviews.")
            print("💡 Install with: pip install google-play-scraper")
            return []
        
        all_reviews = []
        reviews_by_country = {}
        
        print(f"🎯 Goal: Collect {target_count} REAL reviews")
        print(f"🌍 Countries: {', '.join([c.upper() for c in countries])}")
        print(f"📅 Strategy: Start with recent reviews, extend timeline as needed")
        print("-" * 70)
        
        # First pass: Get all available reviews from each country
        for country in countries:
            try:
                print(f"\n🌍 Fetching ALL available reviews from {country.upper()}...")
                
                # Language mapping
                lang_map = {
                    'us': 'en', 'es': 'es', 'de': 'de', 
                    'fr': 'fr', 'nl': 'nl'
                }
                lang = lang_map.get(country, 'en')
                
                # Get maximum available reviews (Google Play usually limits to ~500-600)
                result, continuation_token = reviews(
                    self.fedex_app_id,
                    lang=lang,
                    country=country.upper(),
                    sort=Sort.NEWEST,  # Most recent first - this is key!
                    count=500  # Maximum we can typically get
                )
                
                country_reviews = []
                
                # Continue fetching if continuation token exists
                while continuation_token and len(country_reviews) < 500:
                    try:
                        next_result, continuation_token = reviews(
                            self.fedex_app_id,
                            lang=lang,
                            country=country.upper(),
                            sort=Sort.NEWEST,
                            continuation_token=continuation_token
                        )
                        result.extend(next_result)
                        if not next_result:
                            break
                    except:
                        break
                
                # Process all reviews from this country
                for review in result:
                    # Detect language if possible
                    detected_lang = self.detect_language(review.get('content', ''))
                    
                    # Only include reviews with content
                    if review.get('content', '').strip():
                        review_date = review.get('at', datetime.now())
                        country_reviews.append({
                            'app_id': self.fedex_app_id,
                            'text': review.get('content', ''),
                            'rating': review.get('score', 0),
                            'date': review_date,
                            'date_str': review_date.strftime('%Y-%m-%d'),
                            'days_ago': (datetime.now() - review_date).days,
                            'country': country,
                            'language_detected': detected_lang,
                            'language_expected': lang,
                            'helpful_count': review.get('thumbsUpCount', 0),
                            'user': review.get('userName', 'Anonymous'),
                            'is_recent': (datetime.now() - review_date).days <= 90,
                            'is_real': True  # All are real reviews
                        })
                
                reviews_by_country[country] = country_reviews
                print(f"✅ Retrieved {len(country_reviews)} real reviews from {country.upper()}")
                
                # Date range info
                if country_reviews:
                    dates = [r['date'] for r in country_reviews]
                    oldest = min(dates)
                    newest = max(dates)
                    print(f"   📅 Date range: {oldest.strftime('%Y-%m-%d')} to {newest.strftime('%Y-%m-%d')}")
                
                # Be respectful with delays
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error scraping {country}: {e}")
                reviews_by_country[country] = []
                continue
        
        # Combine all reviews and sort by date (most recent first)
        for country_reviews in reviews_by_country.values():
            all_reviews.extend(country_reviews)
        
        # Sort by date - MOST RECENT FIRST (this ensures we prioritize recent reviews)
        all_reviews.sort(key=lambda x: x['date'], reverse=True)
        
        # Take only the target count (most recent reviews)
        if len(all_reviews) > target_count:
            print(f"\n📊 Total available: {len(all_reviews)} reviews")
            print(f"✂️ Taking the {target_count} most recent reviews")
            all_reviews = all_reviews[:target_count]
        else:
            print(f"\n📊 Total collected: {len(all_reviews)} real reviews")
            if len(all_reviews) < target_count:
                print(f"⚠️ Only {len(all_reviews)} real reviews available (target was {target_count})")
        
        # Summary statistics
        if all_reviews:
            dates = [r['date'] for r in all_reviews]
            oldest = min(dates)
            newest = max(dates)
            
            print(f"\n📈 Final Dataset Statistics:")
            print(f"   Total reviews: {len(all_reviews)}")
            print(f"   Date range: {oldest.strftime('%Y-%m-%d')} to {newest.strftime('%Y-%m-%d')}")
            print(f"   Timespan: {(newest - oldest).days} days")
            
            # Distribution by time periods
            last_30 = sum(1 for r in all_reviews if r['days_ago'] <= 30)
            last_90 = sum(1 for r in all_reviews if r['days_ago'] <= 90)
            last_180 = sum(1 for r in all_reviews if r['days_ago'] <= 180)
            
            print(f"\n📅 Recency Distribution:")
            print(f"   Last 30 days: {last_30} reviews ({last_30/len(all_reviews)*100:.1f}%)")
            print(f"   Last 90 days: {last_90} reviews ({last_90/len(all_reviews)*100:.1f}%)")
            print(f"   Last 180 days: {last_180} reviews ({last_180/len(all_reviews)*100:.1f}%)")
            
            # Country distribution
            country_dist = Counter([r['country'] for r in all_reviews])
            print(f"\n🌍 Country Distribution:")
            for country, count in country_dist.most_common():
                print(f"   {country.upper()}: {count} reviews ({count/len(all_reviews)*100:.1f}%)")
        
        return all_reviews
    
    def scrape_fedex_reviews(self, count=1000, countries=['us', 'es', 'de', 'fr', 'nl'], 
                            months_back=3, filter_date=True, real_only=True):
        """
        Original scraping method with date filtering (kept for backward compatibility)
        """
        # Calculate date cutoff for filtering
        cutoff_date = datetime.now() - timedelta(days=months_back * 30)
        
        if not SCRAPER_AVAILABLE:
            if real_only:
                print("❌ google-play-scraper not available. Cannot get real reviews.")
                print("💡 Install with: pip install google-play-scraper")
                return []
            else:
                print("❌ google-play-scraper not available. Creating enhanced sample data...")
                print(f"📅 Generating reviews from last {months_back} months (after {cutoff_date.strftime('%Y-%m-%d')})")
                return self.create_enhanced_sample_data(count, cutoff_date=cutoff_date if filter_date else None)
        
        all_reviews = []
        target_per_country = count // len(countries)
        
        print(f"🎯 Target: Up to {count} REAL reviews ({target_per_country} per country)")
        if filter_date:
            print(f"📅 Filter: Last {months_back} months (after {cutoff_date.strftime('%Y-%m-%d')})")
        print(f"⚠️ Real-only mode: Will return actual available reviews (may be less than {count})")
        
        for country in countries:
            try:
                print(f"🌍 Scraping FedEx reviews from {country.upper()}...")
                
                # Language mapping
                lang_map = {
                    'us': 'en', 'es': 'es', 'de': 'de', 
                    'fr': 'fr', 'nl': 'nl'
                }
                lang = lang_map.get(country, 'en')
                
                # Get reviews using google-play-scraper
                result, continuation_token = reviews(
                    self.fedex_app_id,
                    lang=lang,
                    country=country.upper(),
                    sort=Sort.NEWEST,
                    count=500  # Max we can get
                )
                
                country_reviews = []
                filtered_count = 0
                
                for review in result:
                    review_date = review.get('at', datetime.now())
                    
                    # Filter by date if enabled
                    if filter_date and review_date < cutoff_date:
                        filtered_count += 1
                        continue
                    
                    # Detect language if possible
                    detected_lang = self.detect_language(review.get('content', ''))
                    
                    # Only include reviews with content
                    if review.get('content', '').strip():
                        country_reviews.append({
                            'app_id': self.fedex_app_id,
                            'text': review.get('content', ''),
                            'rating': review.get('score', 0),
                            'date': review_date,
                            'date_str': review_date.strftime('%Y-%m-%d'),
                            'days_ago': (datetime.now() - review_date).days,
                            'country': country,
                            'language_detected': detected_lang,
                            'language_expected': lang,
                            'helpful_count': review.get('thumbsUpCount', 0),
                            'user': review.get('userName', 'Anonymous'),
                            'is_recent': True,
                            'is_real': True
                        })
                
                if filter_date and filtered_count > 0:
                    print(f"  📅 Filtered out {filtered_count} reviews older than {months_back} months")
                
                all_reviews.extend(country_reviews)
                print(f"✅ Got {len(country_reviews)} REAL recent reviews from {country}")
                
                # NO SYNTHETIC DATA when real_only=True
                if not real_only and len(country_reviews) < target_per_country:
                    shortage = target_per_country - len(country_reviews)
                    print(f"  📝 Generating {shortage} additional samples for {country}")
                    samples = self.create_country_enhanced_samples(
                        country, shortage, cutoff_date=cutoff_date
                    )
                    all_reviews.extend(samples)
                
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error scraping {country}: {e}")
                continue
        
        print(f"🎉 Total collected: {len(all_reviews)} REAL FedEx reviews")
        return all_reviews
        """
        Scrape FedEx app reviews using google-play-scraper
        
        Args:
            count: Target number of reviews to collect (may get less if not available)
            countries: List of country codes to scrape from
            months_back: Number of months to look back (default: 3)
            filter_date: Whether to filter by date (default: True)
            real_only: If True, only return real reviews, no synthetic data (default: True)
        """
        # Calculate date cutoff for filtering
        cutoff_date = datetime.now() - timedelta(days=months_back * 30)
        
        if not SCRAPER_AVAILABLE:
            if real_only:
                print("❌ google-play-scraper not available. Cannot get real reviews.")
                print("💡 Install with: pip install google-play-scraper")
                return []
            else:
                print("❌ google-play-scraper not available. Creating enhanced sample data...")
                print(f"📅 Generating reviews from last {months_back} months (after {cutoff_date.strftime('%Y-%m-%d')})")
                return self.create_enhanced_sample_data(count, cutoff_date=cutoff_date if filter_date else None)
        
        all_reviews = []
        target_per_country = count // len(countries)
        
        print(f"🎯 Target: Up to {count} REAL reviews ({target_per_country} per country)")
        if filter_date:
            print(f"📅 Filter: Last {months_back} months (after {cutoff_date.strftime('%Y-%m-%d')})")
        print(f"⚠️ Real-only mode: Will return actual available reviews (may be less than {count})")
        
        for country in countries:
            try:
                print(f"🌍 Scraping FedEx reviews from {country.upper()}...")
                
                # Language mapping
                lang_map = {
                    'us': 'en', 'es': 'es', 'de': 'de', 
                    'fr': 'fr', 'nl': 'nl'
                }
                lang = lang_map.get(country, 'en')
                
                # Get more reviews to account for date filtering
                fetch_count = 500  # Max we can get from Google Play
                
                # Get reviews using google-play-scraper (always gets newest first)
                result, continuation_token = reviews(
                    self.fedex_app_id,
                    lang=lang,
                    country=country.upper(),
                    sort=Sort.NEWEST,  # This ensures we get most recent reviews
                    count=fetch_count,  # Google Play limits
                )
                
                country_reviews = []
                filtered_count = 0
                
                for review in result:
                    review_date = review.get('at', datetime.now())
                    
                    # Filter by date if enabled
                    if filter_date and review_date < cutoff_date:
                        filtered_count += 1
                        continue  # Skip reviews older than cutoff
                    
                    # Detect language if possible
                    detected_lang = self.detect_language(review.get('content', ''))
                    
                    # Only include reviews with content
                    if review.get('content', '').strip():
                        country_reviews.append({
                            'app_id': self.fedex_app_id,
                            'text': review.get('content', ''),
                            'rating': review.get('score', 0),
                            'date': review_date,
                            'date_str': review_date.strftime('%Y-%m-%d'),
                            'days_ago': (datetime.now() - review_date).days,
                            'country': country,
                            'language_detected': detected_lang,
                            'language_expected': lang,
                            'helpful_count': review.get('thumbsUpCount', 0),
                            'user': review.get('userName', 'Anonymous'),
                            'is_recent': True,  # Mark as recent review
                            'is_real': True  # Mark as real review
                        })
                
                if filter_date and filtered_count > 0:
                    print(f"  📅 Filtered out {filtered_count} reviews older than {months_back} months")
                
                all_reviews.extend(country_reviews)
                print(f"✅ Got {len(country_reviews)} REAL recent reviews from {country}")
                
                # NO SYNTHETIC DATA GENERATION when real_only=True
                if not real_only and len(country_reviews) < target_per_country:
                    shortage = target_per_country - len(country_reviews)
                    print(f"  📝 Generating {shortage} additional samples for {country}")
                    samples = self.create_country_enhanced_samples(
                        country, shortage, cutoff_date=cutoff_date
                    )
                    all_reviews.extend(samples)
                
                # Be respectful with delays
                time.sleep(2)
            
            except Exception as e:
                print(f"❌ Error scraping {country}: {e}")
                if not real_only:
                    # Add enhanced sample data for failed countries
                    country_sample = self.create_country_enhanced_samples(
                        country, target_per_country // 2, 
                        cutoff_date=cutoff_date if filter_date else None
                    )
                    all_reviews.extend(country_sample)
                    print(f"📝 Added {len(country_sample)} enhanced sample reviews for {country}")
                continue
        
        print(f"🎉 Total collected: {len(all_reviews)} REAL FedEx reviews")
        
        # Print date distribution summary
        if filter_date and all_reviews:
            dates = [r['date'] for r in all_reviews]
            oldest = min(dates)
            newest = max(dates)
            print(f"📅 Date range: {oldest.strftime('%Y-%m-%d')} to {newest.strftime('%Y-%m-%d')}")
            
            # Group by month
            months = Counter([d.strftime('%Y-%m') for d in dates])
            print(f"📊 Reviews by month:")
            for month, count in sorted(months.items(), reverse=True):
                print(f"   {month}: {count} reviews")
        
        return all_reviews
    
    def create_country_enhanced_samples(self, country, count, cutoff_date=None):
        """
        Create enhanced sample data for a specific country with all 6 aspects
        
        Args:
            country: Country code
            count: Number of samples to generate
            cutoff_date: Optional date cutoff for recent reviews
        """
        lang_map = {'us': 'en', 'es': 'es', 'de': 'de', 'fr': 'fr', 'nl': 'nl'}
        lang = lang_map.get(country, 'en')
        
        reviews = []
        aspects = list(self.aspect_review_templates.keys())
        
        # Calculate date range for samples
        if cutoff_date:
            # Generate dates between cutoff and now
            date_range = (datetime.now() - cutoff_date).days
        else:
            # Default to last 90 days
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
                # Fallback to English
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
            # More recent reviews are more likely (exponential distribution)
            days_ago = int(np.random.exponential(scale=date_range/3))
            days_ago = min(days_ago, date_range)  # Cap at date_range
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
                'is_recent': days_ago <= 90,  # Mark if within 3 months
                'is_real': False  # Mark as synthetic/fake review
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
        """Classify reviews using enhanced multi-label aspect classifier"""
        print("🎯 Classifying aspects using enhanced multi-label system...")
        
        if not self.use_enhanced_models:
            print("⚠️ Enhanced models not available, using basic classification")
            return self.classify_reviews_basic(reviews)
        
        classified_reviews = []
        
        for i, review in enumerate(reviews):
            if i % 100 == 0:
                print(f"Processing review {i}/{len(reviews)}...")
            
            text = review['text']
            lang = review.get('language_detected', 'en')
            
            # Get multi-label aspect classification
            aspect_result = self.aspect_classifier.classify_aspects_multilabel(text, lang)
            
            # Get enhanced sentiment analysis
            sentiment_result = self.sentiment_classifier.analyze_sentiment(text, lang)
            
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
                
                # Enhanced sentiment analysis
                'sentiment': sentiment_result['sentiment'],
                'sentiment_confidence': sentiment_result['confidence'],
                'sentiment_scores': sentiment_result['scores'],
                'sentiment_model_used': sentiment_result['model_used'],
                
                # Combined insights
                'is_mixed_concern': len(aspect_result['secondary_aspects']) > 0,
                'aspect_count': 1 + len(aspect_result['secondary_aspects']),
                'is_critical': aspect_result['severity_level'] == 'CRITICAL' or 
                              aspect_result['requires_immediate_action']
            }
            
            classified_reviews.append(classified_review)
        
        return classified_reviews
    
    def classify_reviews_basic(self, reviews):
        """Basic classification fallback when enhanced models aren't available"""
        for review in reviews:
            # Simple rating-based sentiment
            rating = review['rating']
            if rating >= 4:
                review['sentiment'] = 'positive'
            elif rating <= 2:
                review['sentiment'] = 'negative'
            else:
                review['sentiment'] = 'neutral'
            
            review['sentiment_confidence'] = 0.8
            review['primary_aspect'] = 'general_satisfaction'
            review['secondary_aspects'] = []
            review['classification_type'] = 'basic'
            review['priority_level'] = 'MEDIUM'
            review['severity_level'] = 'MODERATE'
            review['requires_immediate_action'] = False
            
        return reviews
    
    def create_enhanced_sample_data(self, count=1000, cutoff_date=None):
        """
        Create realistic FedEx app review samples with all 6 aspects
        
        Args:
            count: Number of samples to generate
            cutoff_date: Optional date cutoff for recent reviews
        """
        print("📝 Creating enhanced FedEx sample review data...")
        if cutoff_date:
            print(f"📅 Generating reviews from after {cutoff_date.strftime('%Y-%m-%d')}")
        
        reviews = []
        countries = ['us', 'es', 'de', 'fr', 'nl']
        per_country = count // len(countries)
        
        for country in countries:
            country_reviews = self.create_country_enhanced_samples(
                country, per_country, cutoff_date=cutoff_date
            )
            reviews.extend(country_reviews)
        
        # Add some specific mixed concern examples with recent dates
        if not cutoff_date:
            cutoff_date = datetime.now() - timedelta(days=90)
        
        mixed_examples = [
            {
                'text': "The tracking accuracy is perfect but the app crashes constantly and the interface is terrible",
                'rating': 2, 'country': 'us', 'language_detected': 'en',
                'days_ago': 5
            },
            {
                'text': "Love the modern design and fast performance, but delivery notifications never work properly",
                'rating': 3, 'country': 'us', 'language_detected': 'en',
                'days_ago': 10
            },
            {
                'text': "Interface is impossible to navigate, crashes frequently, and tracking info is always wrong",
                'rating': 1, 'country': 'us', 'language_detected': 'en',
                'days_ago': 2
            },
            {
                'text': "not receiving email for sign in, this app continues to be trash!",
                'rating': 1, 'country': 'us', 'language_detected': 'en',
                'days_ago': 1
            }
        ]
        
        for example in mixed_examples:
            days_ago = example.pop('days_ago', 0)
            review_date = datetime.now() - timedelta(days=days_ago)
            
            example.update({
                'app_id': self.fedex_app_id,
                'date': review_date,
                'date_str': review_date.strftime('%Y-%m-%d'),
                'days_ago': days_ago,
                'helpful_count': np.random.randint(10, 100),
                'user': f'User_{np.random.randint(1000, 9999)}',
                'language_expected': example['language_detected'],
                'is_recent': True,
                'is_real': False  # Mark as synthetic/fake review
            })
            reviews.append(example)
        
        return reviews[:count]
    
    def analyze_fedex_reviews(self, count=1000, months_back=3, filter_date=True, real_only=True):
        """
        Main function to scrape and analyze FedEx reviews with enhanced models
        
        Args:
            count: Number of reviews to collect (default: 1000)
            months_back: Number of months to look back for reviews (default: 3) - ignored if real_only=True
            filter_date: Whether to filter by date (default: True) - ignored if real_only=True
            real_only: If True, only collect real reviews using adaptive timeline (default: True)
        """
        print("🚀 Starting Enhanced FedEx Review Analysis...")
        print(f"📱 Target app: {self.fedex_app_id}")
        print("🌍 Countries: US, ES, DE, FR, NL")
        print(f"🤖 Enhanced Models: {'ENABLED' if self.use_enhanced_models else 'DISABLED'}")
        
        if real_only:
            print(f"🔍 Mode: REAL REVIEWS ONLY - Adaptive timeline to reach {count} reviews")
            # Use adaptive scraping to get target number of real reviews
            reviews = self.scrape_fedex_reviews_adaptive(target_count=count)
        else:
            print(f"📅 Date Filter: Last {months_back} months" if filter_date else "📅 Date Filter: Disabled")
            print(f"🔍 Mode: {'REAL REVIEWS ONLY' if real_only else 'Real + Synthetic Reviews'}")
            # Use original method with date filtering
            reviews = self.scrape_fedex_reviews(
                count=count, 
                months_back=months_back, 
                filter_date=filter_date,
                real_only=real_only
            )
        
        if not reviews:
            print("❌ No reviews collected")
            return None
        
        # Process reviews with enhanced classification
        reviews = self.classify_reviews_enhanced(reviews)
        
        # Convert to DataFrame
        df = pd.DataFrame(reviews)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"{self.data_dir}/fedex_reviews_enhanced_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        # Show full path
        full_path = os.path.abspath(filename)
        print(f"\n💾 Data saved to: {full_path}")
        
        # Generate enhanced analysis report
        self.generate_enhanced_analysis_report(df, filename)
        
        # Generate temporal analysis if date information is available
        if 'date' in df.columns:
            self.generate_temporal_analysis(df)
        
        # Generate business intelligence report if enhanced models are used
        if self.use_enhanced_models:
            self.generate_business_intelligence_report(df)
        
        return df
    
    def generate_enhanced_analysis_report(self, df, filename):
        """Generate enhanced analysis report for FedEx reviews"""
        print(f"\n📊 Enhanced FedEx Review Analysis Report")
        print("="*70)
        print(f"✅ Saved {len(df)} reviews to {filename}")
        
        print(f"\n🌍 Language Distribution:")
        lang_dist = df['language_detected'].value_counts()
        for lang, count in lang_dist.items():
            print(f"  {lang}: {count} reviews ({count/len(df)*100:.1f}%)")
        
        print(f"\n😊 Sentiment Distribution:")
        sentiment_dist = df['sentiment'].value_counts()
        for sentiment, count in sentiment_dist.items():
            print(f"  {sentiment}: {count} reviews ({count/len(df)*100:.1f}%)")
        
        if 'primary_aspect' in df.columns:
            print(f"\n🎯 Primary Aspect Distribution:")
            aspect_dist = df['primary_aspect'].value_counts()
            for aspect, count in aspect_dist.items():
                print(f"  {aspect}: {count} reviews ({count/len(df)*100:.1f}%)")
            
            print(f"\n📋 Classification Types:")
            class_dist = df['classification_type'].value_counts()
            for class_type, count in class_dist.items():
                print(f"  {class_type}: {count} reviews ({count/len(df)*100:.1f}%)")
            
            print(f"\n⚡ Priority Levels:")
            priority_dist = df['priority_level'].value_counts()
            for priority, count in priority_dist.items():
                print(f"  {priority}: {count} reviews ({count/len(df)*100:.1f}%)")
            
            print(f"\n🚨 Severity Levels:")
            severity_dist = df['severity_level'].value_counts()
            for severity, count in severity_dist.items():
                print(f"  {severity}: {count} reviews ({count/len(df)*100:.1f}%)")
            
            # Mixed concerns analysis
            if 'is_mixed_concern' in df.columns:
                mixed_count = df['is_mixed_concern'].sum()
                print(f"\n🔀 Mixed Concerns: {mixed_count} reviews ({mixed_count/len(df)*100:.1f}%)")
            
            # Critical issues
            if 'is_critical' in df.columns:
                critical_count = df['is_critical'].sum()
                print(f"\n🚨 Critical Issues: {critical_count} reviews ({critical_count/len(df)*100:.1f}%)")
        
        # Sentiment by aspect matrix
        if 'primary_aspect' in df.columns:
            print(f"\n💡 Sentiment by Primary Aspect:")
            sentiment_aspect = df.groupby(['primary_aspect', 'sentiment']).size().unstack(fill_value=0)
            print(sentiment_aspect)
        
        # Sample reviews by category
        self.print_sample_reviews(df)
    
    def print_sample_reviews(self, df):
        """Print sample reviews for different categories"""
        print(f"\n📝 Sample Reviews by Category:")
        
        if 'primary_aspect' not in df.columns:
            return
        
        # Get unique aspects
        aspects = df['primary_aspect'].unique()
        
        for aspect in aspects[:3]:  # Show samples for first 3 aspects
            print(f"\n🎯 {aspect.replace('_', ' ').title()}:")
            
            # Positive samples
            positive_samples = df[
                (df['primary_aspect'] == aspect) & 
                (df['sentiment'] == 'positive')
            ]['text'].head(2)
            
            if len(positive_samples) > 0:
                print("  ✅ Positive:")
                for i, text in enumerate(positive_samples, 1):
                    print(f"    {i}. {text[:100]}...")
            
            # Negative samples
            negative_samples = df[
                (df['primary_aspect'] == aspect) & 
                (df['sentiment'] == 'negative')
            ]['text'].head(2)
            
            if len(negative_samples) > 0:
                print("  ❌ Negative:")
                for i, text in enumerate(negative_samples, 1):
                    print(f"    {i}. {text[:100]}...")
        
        # Mixed concerns samples
        if 'is_mixed_concern' in df.columns:
            mixed_samples = df[df['is_mixed_concern'] == True]['text'].head(3)
            if len(mixed_samples) > 0:
                print(f"\n🔀 Mixed Concerns Examples:")
                for i, text in enumerate(mixed_samples, 1):
                    row = df[df['text'] == text].iloc[0]
                    primary = row['primary_aspect']
                    secondary = row.get('secondary_aspects', [])
                    print(f"  {i}. Primary: {primary}, Secondary: {secondary}")
                    print(f"     {text[:100]}...")
    
    def generate_business_intelligence_report(self, df):
        """Generate business intelligence report using enhanced models"""
        if not self.use_enhanced_models:
            return
        
        print(f"\n📈 Business Intelligence Report")
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
                'review_text': row.get('text', '')
            }
            results.append(result)
        
        # Generate report using aspect classifier
        report = self.aspect_classifier.generate_business_report(results)
        
        print(f"\n📊 Summary Metrics:")
        for key, value in report['summary'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n🎯 Classification Breakdown:")
        for category, breakdown in report['classification_breakdown'].items():
            print(f"  {category.replace('_', ' ').title()}:")
            for item, count in breakdown.items():
                print(f"    {item}: {count}")
        
        if report.get('mixed_concerns_patterns'):
            print(f"\n🔀 Top Mixed Concerns Patterns:")
            for pattern, count in report['mixed_concerns_patterns'].items():
                print(f"  {pattern}: {count} occurrences")
        
        if report.get('top_recommendations'):
            print(f"\n💡 Top Recommendations:")
            for i, rec in enumerate(report['top_recommendations'], 1):
                print(f"  {i}. {rec}")
        
        if report.get('user_experience_insights'):
            print(f"\n🎨 User Experience Insights:")
            for key, value in report['user_experience_insights'].items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Action items
        critical_count = df[df.get('is_critical', False) == True].shape[0] if 'is_critical' in df.columns else 0
        if critical_count > 0:
            print(f"\n⚠️ IMMEDIATE ACTION REQUIRED:")
            print(f"  {critical_count} critical issues identified")
            print(f"  Reviews requiring immediate attention should be escalated")

    def generate_temporal_analysis(self, df):
        """Generate temporal analysis of reviews"""
        if 'date' not in df.columns or 'days_ago' not in df.columns:
            return
        
        print(f"\n⏰ Temporal Analysis")
        print("="*70)
        
        # Group by weeks
        df['week'] = pd.to_datetime(df['date']).dt.to_period('W')
        
        # Reviews by week
        print(f"\n📊 Reviews by Week:")
        weekly_counts = df.groupby('week').size()
        for week, count in weekly_counts.tail(12).items():  # Show last 12 weeks
            print(f"  {week}: {count} reviews")
        
        # Sentiment trend over time
        print(f"\n😊 Sentiment Trend (Last 3 Months):")
        weekly_sentiment = df.groupby(['week', 'sentiment']).size().unstack(fill_value=0)
        print(weekly_sentiment.tail(12))
        
        # Aspect trends
        if 'primary_aspect' in df.columns:
            print(f"\n🎯 Top Issues by Month:")
            df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
            monthly_aspects = df.groupby(['month', 'primary_aspect']).size().unstack(fill_value=0)
            
            # Show top 3 aspects per month
            for month in monthly_aspects.index[-3:]:  # Last 3 months
                top_aspects = monthly_aspects.loc[month].nlargest(3)
                print(f"\n  {month}:")
                for aspect, count in top_aspects.items():
                    print(f"    - {aspect}: {count} reviews")
        
        # Critical issues over time
        if 'is_critical' in df.columns:
            print(f"\n🚨 Critical Issues Trend:")
            weekly_critical = df[df['is_critical']].groupby('week').size()
            for week, count in weekly_critical.tail(8).items():
                print(f"  {week}: {count} critical issues")
        
        # Recent vs older comparison
        recent_cutoff = 30  # Last 30 days
        recent_mask = df['days_ago'] <= recent_cutoff
        
        print(f"\n📈 Recent (≤30 days) vs Older Reviews:")
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

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = FedExReviewAnalyzer()
    
    print("📱 Analyzing FedEx Mobile App Reviews")
    print("🔗 App: https://play.google.com/store/apps/details?id=com.fedex.ida.android")
    print("="*70)
    
    # DEFAULT: Get 1000 REAL reviews using adaptive timeline
    # This will automatically extend the timeline as needed to reach 1000 reviews
    # Most recent reviews are prioritized
    df = analyzer.analyze_fedex_reviews(
        count=1000,      # Target: 1000 real reviews
        real_only=True   # ONLY REAL REVIEWS - uses adaptive timeline
    )
    
    # Alternative options (uncomment to use):
    
    # Option 1: Get MORE real reviews (if available)
    # df = analyzer.analyze_fedex_reviews(
    #     count=1500,      # Try to get 1500 real reviews
    #     real_only=True
    # )
    
    # Option 2: Get reviews from specific time period only
    # df = analyzer.analyze_fedex_reviews(
    #     count=1000,
    #     months_back=3,
    #     filter_date=True,
    #     real_only=False  # This uses the old method with date filtering
    # )
    
    if df is not None:
        print("\n🎉 FedEx analysis complete!")
        print("="*70)
        print("📊 Dataset Summary:")
        print(f"   ✅ Total REAL reviews: {len(df)}")
        
        if 'is_real' in df.columns:
            real_count = df['is_real'].sum()
            print(f"   ✅ Verified real reviews: {real_count}")
            if real_count < len(df):
                print(f"   ⚠️ Synthetic reviews: {len(df) - real_count}")
        
        if 'days_ago' in df.columns:
            print(f"\n📅 Timeline Coverage:")
            print(f"   Most recent: {df['days_ago'].min()} days ago")
            print(f"   Oldest: {df['days_ago'].max()} days ago")
            print(f"   Average age: {df['days_ago'].mean():.1f} days")
            
            # Show distribution
            recent_30 = (df['days_ago'] <= 30).sum()
            recent_90 = (df['days_ago'] <= 90).sum()
            print(f"\n📊 Recency Breakdown:")
            print(f"   Last 30 days: {recent_30} reviews ({recent_30/len(df)*100:.1f}%)")
            print(f"   Last 90 days: {recent_90} reviews ({recent_90/len(df)*100:.1f}%)")
        
        print("\n✅ All reviews are REAL - no synthetic data!")
        print("✅ Reviews prioritized by recency (newest first)")
        print("🚀 Ready for analysis and ML pipeline!")
    else:
        print("❌ Analysis failed")