import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import re

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

class FedExReviewAnalyzer:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # FedEx app ID for different app stores
        self.fedex_app_id = "com.fedex.ida.android"
        
        # Logistics-specific keywords for aspect detection
        self.aspect_keywords = {
            'product_quality': {
                'en': [
                    'tracking', 'delivery', 'notification', 'accurate', 'reliable', 'performance',
                    'crash', 'bug', 'error', 'slow', 'fast', 'update', 'sync', 'connection',
                    'package', 'status', 'location', 'estimate', 'delay', 'on time'
                ],
                'es': [
                    'seguimiento', 'entrega', 'notificación', 'preciso', 'confiable', 'rendimiento',
                    'fallo', 'error', 'lento', 'rápido', 'actualización', 'sincronización',
                    'paquete', 'estado', 'ubicación', 'estimación', 'retraso'
                ],
                'de': [
                    'verfolgung', 'lieferung', 'benachrichtigung', 'genau', 'zuverlässig', 'leistung',
                    'absturz', 'fehler', 'langsam', 'schnell', 'aktualisierung', 'synchronisation',
                    'paket', 'status', 'standort', 'schätzung', 'verspätung'
                ],
                'fr': [
                    'suivi', 'livraison', 'notification', 'précis', 'fiable', 'performance',
                    'plantage', 'erreur', 'lent', 'rapide', 'mise à jour', 'synchronisation',
                    'colis', 'statut', 'emplacement', 'estimation', 'retard'
                ],
                'nl': [
                    'tracking', 'levering', 'melding', 'nauwkeurig', 'betrouwbaar', 'prestatie',
                    'crash', 'fout', 'traag', 'snel', 'update', 'synchronisatie',
                    'pakket', 'status', 'locatie', 'schatting', 'vertraging'
                ]
            },
            'user_experience': {
                'en': [
                    'easy', 'difficult', 'interface', 'design', 'navigate', 'intuitive', 'confusing',
                    'user friendly', 'simple', 'complex', 'menu', 'button', 'screen', 'layout',
                    'scan', 'barcode', 'search', 'find', 'organize'
                ],
                'es': [
                    'fácil', 'difícil', 'interfaz', 'diseño', 'navegar', 'intuitivo', 'confuso',
                    'amigable', 'simple', 'complejo', 'menú', 'botón', 'pantalla', 'diseño',
                    'escanear', 'código', 'buscar', 'encontrar', 'organizar'
                ],
                'de': [
                    'einfach', 'schwierig', 'benutzeroberfläche', 'design', 'navigieren', 'intuitiv',
                    'verwirrend', 'benutzerfreundlich', 'einfach', 'komplex', 'menü', 'taste',
                    'bildschirm', 'layout', 'scannen', 'strichcode', 'suchen'
                ],
                'fr': [
                    'facile', 'difficile', 'interface', 'design', 'naviguer', 'intuitif', 'confus',
                    'convivial', 'simple', 'complexe', 'menu', 'bouton', 'écran', 'mise en page',
                    'scanner', 'code barre', 'chercher', 'trouver'
                ],
                'nl': [
                    'makkelijk', 'moeilijk', 'interface', 'ontwerp', 'navigeren', 'intuïtief',
                    'verwarrend', 'gebruiksvriendelijk', 'simpel', 'complex', 'menu', 'knop',
                    'scherm', 'indeling', 'scannen', 'streepjescode', 'zoeken'
                ]
            }
        }
    
    def scrape_fedex_reviews(self, count=500, countries=['us', 'es', 'de', 'fr', 'nl']):
        """Scrape FedEx app reviews using google-play-scraper"""
        if not SCRAPER_AVAILABLE:
            print("❌ google-play-scraper not available. Creating sample data...")
            return self.create_fedex_sample_data()
        
        all_reviews = []
        target_per_country = count // len(countries)
        
        print(f"🎯 Target: {count} total reviews ({target_per_country} per country)")
        
        for country in countries:
            try:
                print(f"🌐 Scraping FedEx reviews from {country.upper()}...")
                
                # Language mapping
                lang_map = {
                    'us': 'en', 'es': 'es', 'de': 'de', 
                    'fr': 'fr', 'nl': 'nl'
                }
                lang = lang_map.get(country, 'en')
                
                # Get reviews using google-play-scraper
                # Note: google-play-scraper doesn't have country-specific scraping,
                # so we'll get reviews and filter by language
                result, continuation_token = reviews(
                    self.fedex_app_id,
                    lang=lang,
                    country=country.upper(),
                    sort=Sort.NEWEST,
                    count=target_per_country * 2,  # Get more to account for filtering
                )
                
                country_reviews = []
                for review in result:
                    if len(country_reviews) >= target_per_country:
                        break
                    
                    # Detect language if possible
                    detected_lang = self.detect_language(review.get('content', ''))
                    
                    # Only include reviews with content
                    if review.get('content', '').strip():
                        country_reviews.append({
                            'app_id': self.fedex_app_id,
                            'text': review.get('content', ''),
                            'rating': review.get('score', 0),
                            'date': review.get('at', datetime.now()),
                            'country': country,
                            'language_detected': detected_lang,
                            'language_expected': lang,
                            'helpful_count': review.get('thumbsUpCount', 0),
                            'user': review.get('userName', 'Anonymous')
                        })
                
                all_reviews.extend(country_reviews)
                print(f"✅ Got {len(country_reviews)} reviews from {country}")
                
                # Be respectful with delays
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error scraping {country}: {e}")
                # Add some sample data for failed countries
                country_sample = self.create_country_sample_data(country, target_per_country // 2)
                all_reviews.extend(country_sample)
                print(f"📝 Added {len(country_sample)} sample reviews for {country}")
                continue
        
        print(f"🎉 Total collected: {len(all_reviews)} FedEx reviews")
        return all_reviews
    
    def create_country_sample_data(self, country, count):
        """Create sample data for a specific country"""
        lang_map = {'us': 'en', 'es': 'es', 'de': 'de', 'fr': 'fr', 'nl': 'nl'}
        lang = lang_map.get(country, 'en')
        
        sample_reviews_by_lang = {
            'en': [
                "Tracking is very accurate, always shows correct package location",
                "App crashes when trying to track multiple packages",
                "Very easy to use, intuitive interface for package management",
                "Interface is confusing, hard to find tracking information"
            ],
            'es': [
                "El seguimiento es muy preciso, siempre muestra la ubicación correcta",
                "La aplicación se cierra cuando trato de rastrear varios paquetes", 
                "Muy fácil de usar, interfaz intuitiva para gestionar paquetes",
                "La interfaz es confusa, difícil encontrar información de seguimiento"
            ],
            'de': [
                "Verfolgung ist sehr genau, zeigt immer den korrekten Paketstandort",
                "App stürzt ab beim Verfolgen mehrerer Pakete",
                "Sehr einfach zu bedienen, intuitive Benutzeroberfläche"
            ],
            'fr': [
                "Le suivi est très précis, montre toujours l'emplacement correct",
                "Interface très confuse, difficile de trouver les informations"
            ],
            'nl': [
                "Tracking is zeer nauwkeurig, toont altijd de juiste locatie",
                "Interface is verwarrend, moeilijk om tracking info te vinden"
            ]
        }
        
        texts = sample_reviews_by_lang.get(lang, sample_reviews_by_lang['en'])
        reviews = []
        
        for i in range(min(count, len(texts))):
            text = texts[i % len(texts)]
            rating = np.random.choice([1, 2, 4, 5], p=[0.2, 0.2, 0.3, 0.3])
            
            reviews.append({
                'app_id': self.fedex_app_id,
                'text': text,
                'rating': rating,
                'date': datetime.now(),
                'country': country,
                'language_detected': lang,
                'language_expected': lang,
                'helpful_count': np.random.randint(0, 10),
                'user': f'SampleUser_{np.random.randint(1000, 9999)}'
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
    
    def classify_fedex_aspects(self, reviews):
        """Classify reviews into product quality vs user experience for FedEx"""
        print("🎯 Classifying aspects for FedEx reviews...")
        
        for review in reviews:
            text = review['text'].lower()
            lang = review.get('language_detected', 'en')
            
            # Default to English keywords if language not supported
            if lang not in self.aspect_keywords['product_quality']:
                lang = 'en'
            
            # Count keyword matches
            quality_score = 0
            ux_score = 0
            
            # Product quality keywords
            for keyword in self.aspect_keywords['product_quality'][lang]:
                if keyword in text:
                    quality_score += 1
            
            # User experience keywords  
            for keyword in self.aspect_keywords['user_experience'][lang]:
                if keyword in text:
                    ux_score += 1
            
            # Classify aspect
            if quality_score > ux_score:
                review['aspect'] = 'product_quality'
                review['aspect_confidence'] = quality_score / (quality_score + ux_score + 1)
            elif ux_score > quality_score:
                review['aspect'] = 'user_experience'
                review['aspect_confidence'] = ux_score / (quality_score + ux_score + 1)
            else:
                review['aspect'] = 'general'
                review['aspect_confidence'] = 0.5
            
            # Additional FedEx-specific classifications
            review['mentions_tracking'] = any(word in text for word in ['track', 'seguimiento', 'verfolgung', 'suivi'])
            review['mentions_delivery'] = any(word in text for word in ['delivery', 'entrega', 'lieferung', 'livraison'])
            review['mentions_interface'] = any(word in text for word in ['interface', 'interfaz', 'benutzeroberfläche'])
        
        return reviews
    
    def create_sentiment_labels(self, reviews):
        """Convert ratings to sentiment labels"""
        for review in reviews:
            rating = review['rating']
            if rating >= 4:
                review['sentiment'] = 'positive'
            elif rating <= 2:
                review['sentiment'] = 'negative'
            else:
                review['sentiment'] = 'neutral'
        
        return reviews
    
    def create_fedex_sample_data(self):
        """Create realistic FedEx app review samples as fallback"""
        print("📝 Creating FedEx sample review data...")
        
        sample_reviews = [
            # English - Product Quality
            {"text": "Tracking is very accurate, always shows correct package location", "rating": 5, "country": "us", "language_detected": "en"},
            {"text": "App crashes when trying to track multiple packages", "rating": 1, "country": "us", "language_detected": "en"},
            {"text": "Delivery notifications work perfectly, very reliable", "rating": 5, "country": "us", "language_detected": "en"},
            {"text": "Tracking information is often delayed or wrong", "rating": 2, "country": "us", "language_detected": "en"},
            
            # English - User Experience
            {"text": "Very easy to use, intuitive interface for package management", "rating": 5, "country": "us", "language_detected": "en"},
            {"text": "Interface is confusing, hard to find tracking information", "rating": 2, "country": "us", "language_detected": "en"},
            {"text": "Scanning barcodes is simple and works great", "rating": 4, "country": "us", "language_detected": "en"},
            {"text": "Menu layout is terrible, buttons are hard to find", "rating": 1, "country": "us", "language_detected": "en"},
            
            # Spanish
            {"text": "El seguimiento es muy preciso, siempre muestra la ubicación correcta", "rating": 5, "country": "es", "language_detected": "es"},
            {"text": "La aplicación se cierra cuando trato de rastrear varios paquetes", "rating": 1, "country": "es", "language_detected": "es"},
            {"text": "Muy fácil de usar, interfaz intuitiva para gestionar paquetes", "rating": 5, "country": "es", "language_detected": "es"},
            {"text": "La interfaz es confusa, difícil encontrar información de seguimiento", "rating": 2, "country": "es", "language_detected": "es"},
            
            # German
            {"text": "Verfolgung ist sehr genau, zeigt immer den korrekten Paketstandort", "rating": 5, "country": "de", "language_detected": "de"},
            {"text": "App stürzt ab beim Verfolgen mehrerer Pakete", "rating": 1, "country": "de", "language_detected": "de"},
            {"text": "Sehr einfach zu bedienen, intuitive Benutzeroberfläche", "rating": 5, "country": "de", "language_detected": "de"},
            
            # French
            {"text": "Le suivi est très précis, montre toujours l'emplacement correct", "rating": 5, "country": "fr", "language_detected": "fr"},
            {"text": "Interface très confuse, difficile de trouver les informations", "rating": 2, "country": "fr", "language_detected": "fr"},
            
            # Dutch
            {"text": "Tracking is zeer nauwkeurig, toont altijd de juiste locatie", "rating": 5, "country": "nl", "language_detected": "nl"},
            {"text": "Interface is verwarrend, moeilijk om tracking info te vinden", "rating": 2, "country": "nl", "language_detected": "nl"},
        ]
        
        # Add metadata
        for review in sample_reviews:
            review.update({
                'app_id': self.fedex_app_id,
                'date': datetime.now(),
                'helpful_count': np.random.randint(0, 20),
                'user': f'User_{np.random.randint(1000, 9999)}'
            })
        
        return sample_reviews
    
    def analyze_fedex_reviews(self, count=500):
        """Main function to scrape and analyze FedEx reviews"""
        print("🚀 Starting FedEx review analysis...")
        print(f"📱 Target app: {self.fedex_app_id}")
        print("🌍 Countries: US, ES, DE, FR, NL")
        
        # Scrape reviews
        reviews = self.scrape_fedex_reviews(count=count)
        
        if not reviews:
            print("❌ No reviews collected")
            return None
        
        # Process reviews
        reviews = self.create_sentiment_labels(reviews)
        reviews = self.classify_fedex_aspects(reviews)
        
        # Convert to DataFrame
        df = pd.DataFrame(reviews)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"{self.data_dir}/fedex_reviews_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        # Generate analysis report
        self.generate_analysis_report(df, filename)
        
        return df
    
    def generate_analysis_report(self, df, filename):
        """Generate analysis report for FedEx reviews"""
        print(f"\n📊 FedEx Review Analysis Report")
        print("="*50)
        print(f"✅ Saved {len(df)} reviews to {filename}")
        
        print(f"\n🌍 Language Distribution:")
        lang_dist = df['language_detected'].value_counts()
        for lang, count in lang_dist.items():
            print(f"  {lang}: {count} reviews ({count/len(df)*100:.1f}%)")
        
        print(f"\n😊 Sentiment Distribution:")
        sentiment_dist = df['sentiment'].value_counts()
        for sentiment, count in sentiment_dist.items():
            print(f"  {sentiment}: {count} reviews ({count/len(df)*100:.1f}%)")
        
        print(f"\n🎯 Aspect Distribution:")
        aspect_dist = df['aspect'].value_counts()
        for aspect, count in aspect_dist.items():
            print(f"  {aspect}: {count} reviews ({count/len(df)*100:.1f}%)")
        
        print(f"\n📦 FedEx-Specific Insights:")
        print(f"  Reviews mentioning tracking: {df['mentions_tracking'].sum()}")
        print(f"  Reviews mentioning delivery: {df['mentions_delivery'].sum()}")
        print(f"  Reviews mentioning interface: {df['mentions_interface'].sum()}")
        
        # Sentiment by aspect
        print(f"\n💡 Sentiment by Aspect:")
        sentiment_aspect = df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
        print(sentiment_aspect)
        
        print(f"\n🔍 Sample Reviews by Category:")
        
        # Product quality examples
        quality_positive = df[(df['aspect'] == 'product_quality') & (df['sentiment'] == 'positive')]['text'].iloc[:2]
        quality_negative = df[(df['aspect'] == 'product_quality') & (df['sentiment'] == 'negative')]['text'].iloc[:2]
        
        print(f"\n✅ Positive Product Quality:")
        for i, text in enumerate(quality_positive, 1):
            print(f"  {i}. {text[:100]}...")
        
        print(f"\n❌ Negative Product Quality:")
        for i, text in enumerate(quality_negative, 1):
            print(f"  {i}. {text[:100]}...")
        
        # User experience examples
        ux_positive = df[(df['aspect'] == 'user_experience') & (df['sentiment'] == 'positive')]['text'].iloc[:2]
        ux_negative = df[(df['aspect'] == 'user_experience') & (df['sentiment'] == 'negative')]['text'].iloc[:2]
        
        print(f"\n✅ Positive User Experience:")
        for i, text in enumerate(ux_positive, 1):
            print(f"  {i}. {text[:100]}...")
        
        print(f"\n❌ Negative User Experience:")
        for i, text in enumerate(ux_negative, 1):
            print(f"  {i}. {text[:100]}...")

if __name__ == "__main__":
    analyzer = FedExReviewAnalyzer()
    
    # Analyze FedEx reviews
    print("📱 Analyzing FedEx Mobile App Reviews")
    print("🔗 App: https://play.google.com/store/apps/details?id=com.fedex.ida.android")
    
    df = analyzer.analyze_fedex_reviews(count=500)
    
    if df is not None:
        print("\n🎉 FedEx analysis complete!")
        print("📁 Data ready for machine learning pipeline!")
    else:
        print("❌ Analysis failed")