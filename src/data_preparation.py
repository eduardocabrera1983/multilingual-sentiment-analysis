import pandas as pd
import numpy as np
from datasets import load_dataset
import os
from langdetect import detect
import warnings
warnings.filterwarnings('ignore')

class DatasetPreparer:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_amazon_reviews(self):
        """Download Amazon multilingual reviews dataset"""
        print("📥 Downloading Amazon multilingual reviews...")
        
        # Load Amazon reviews in multiple languages
        try:
            # English reviews
            dataset_en = load_dataset("amazon_reviews_multi", "en", split="train[:5000]")
            df_en = pd.DataFrame(dataset_en)
            df_en['language'] = 'en'
            
            # German reviews  
            dataset_de = load_dataset("amazon_reviews_multi", "de", split="train[:2000]")
            df_de = pd.DataFrame(dataset_de)
            df_de['language'] = 'de'
            
            # Spanish reviews
            dataset_es = load_dataset("amazon_reviews_multi", "es", split="train[:2000]")
            df_es = pd.DataFrame(dataset_es)
            df_es['language'] = 'es'
            
            # Combine datasets
            df_combined = pd.concat([df_en, df_de, df_es], ignore_index=True)
            
            # Select relevant columns
            df_processed = df_combined[['review_body', 'stars', 'language']].copy()
            df_processed = df_processed.dropna()
            
            # Create sentiment labels
            df_processed['sentiment'] = df_processed['stars'].apply(self._stars_to_sentiment)
            
            # Save processed data
            df_processed.to_csv(f"{self.data_dir}/amazon_reviews_multilingual.csv", index=False)
            print(f"✅ Saved {len(df_processed)} Amazon reviews to {self.data_dir}/amazon_reviews_multilingual.csv")
            
            return df_processed
            
        except Exception as e:
            print(f"❌ Error downloading Amazon dataset: {e}")
            return self._create_sample_data()
    
    def create_sample_business_data(self):
        """Create realistic sample business feedback data"""
        print("📝 Creating sample business feedback data...")
        
        # Sample feedback in multiple languages focusing on product quality and user experience
        sample_data = [
            # English - Product Quality
            {"text": "The product quality is excellent, very durable and well-made", "language": "en", "aspect": "product_quality", "sentiment": "positive"},
            {"text": "Poor build quality, broke after one week of use", "language": "en", "aspect": "product_quality", "sentiment": "negative"},
            {"text": "Average quality for the price, nothing special", "language": "en", "aspect": "product_quality", "sentiment": "neutral"},
            
            # English - User Experience
            {"text": "Very easy to use, intuitive interface and smooth experience", "language": "en", "aspect": "user_experience", "sentiment": "positive"},
            {"text": "Confusing interface, hard to navigate and find features", "language": "en", "aspect": "user_experience", "sentiment": "negative"},
            {"text": "Interface is okay, takes some time to learn", "language": "en", "aspect": "user_experience", "sentiment": "neutral"},
            
            # Spanish - Product Quality
            {"text": "La calidad del producto es excelente, muy duradero", "language": "es", "aspect": "product_quality", "sentiment": "positive"},
            {"text": "Mala calidad, se rompió después de una semana", "language": "es", "aspect": "product_quality", "sentiment": "negative"},
            {"text": "Calidad promedio por el precio", "language": "es", "aspect": "product_quality", "sentiment": "neutral"},
            
            # Spanish - User Experience
            {"text": "Muy fácil de usar, interfaz intuitiva", "language": "es", "aspect": "user_experience", "sentiment": "positive"},
            {"text": "Interfaz confusa, difícil de navegar", "language": "es", "aspect": "user_experience", "sentiment": "negative"},
            
            # German - Product Quality
            {"text": "Ausgezeichnete Produktqualität, sehr langlebig", "language": "de", "aspect": "product_quality", "sentiment": "positive"},
            {"text": "Schlechte Qualität, nach einer Woche kaputt", "language": "de", "aspect": "product_quality", "sentiment": "negative"},
            
            # German - User Experience
            {"text": "Sehr benutzerfreundlich, intuitive Bedienung", "language": "de", "aspect": "user_experience", "sentiment": "positive"},
            {"text": "Verwirrende Benutzeroberfläche, schwer zu navigieren", "language": "de", "aspect": "user_experience", "sentiment": "negative"},
            
            # French - Product Quality
            {"text": "Excellente qualité de produit, très durable", "language": "fr", "aspect": "product_quality", "sentiment": "positive"},
            {"text": "Mauvaise qualité, cassé après une semaine", "language": "fr", "aspect": "product_quality", "sentiment": "negative"},
            
            # French - User Experience
            {"text": "Très facile à utiliser, interface intuitive", "language": "fr", "aspect": "user_experience", "sentiment": "positive"},
            {"text": "Interface confuse, difficile à naviguer", "language": "fr", "aspect": "user_experience", "sentiment": "negative"},
        ]
        
        # Create DataFrame
        df_sample = pd.DataFrame(sample_data)
        
        # Save sample data
        df_sample.to_csv(f"{self.data_dir}/sample_business_feedback.csv", index=False)
        print(f"✅ Created sample business feedback with {len(df_sample)} entries")
        
        return df_sample
    
    def _stars_to_sentiment(self, stars):
        """Convert star ratings to sentiment labels"""
        if stars >= 4:
            return "positive"
        elif stars <= 2:
            return "negative"
        else:
            return "neutral"
    
    def _create_sample_data(self):
        """Fallback: create sample data if download fails"""
        print("📝 Creating fallback sample data...")
        
        sample_reviews = [
            {"review_body": "Great product, highly recommended!", "stars": 5, "language": "en"},
            {"review_body": "Terrible quality, waste of money", "stars": 1, "language": "en"},
            {"review_body": "Excelente producto, muy recomendado", "stars": 5, "language": "es"},
            {"review_body": "Mala calidad, pérdida de dinero", "stars": 1, "language": "es"},
            {"review_body": "Großartiges Produkt, sehr empfehlenswert", "stars": 5, "language": "de"},
            {"review_body": "Schlechte Qualität, Geldverschwendung", "stars": 1, "language": "de"},
        ]
        
        df = pd.DataFrame(sample_reviews)
        df['sentiment'] = df['stars'].apply(self._stars_to_sentiment)
        df.to_csv(f"{self.data_dir}/sample_data.csv", index=False)
        
        return df
    
    def prepare_all_datasets(self):
        """Prepare all datasets for the project"""
        print("🚀 Starting data preparation...")
        
        # Download Amazon reviews
        amazon_df = self.download_amazon_reviews()
        
        # Create business sample data
        business_df = self.create_sample_business_data()
        
        print(f"\n📊 Data Summary:")
        print(f"Amazon reviews: {len(amazon_df)} samples")
        print(f"Business feedback: {len(business_df)} samples")
        print(f"Languages covered: {business_df['language'].unique()}")
        print(f"Aspects covered: {business_df['aspect'].unique()}")
        
        return amazon_df, business_df

if __name__ == "__main__":
    # Initialize data preparer
    preparer = DatasetPreparer()
    
    # Prepare all datasets
    amazon_data, business_data = preparer.prepare_all_datasets()
    
    print("\n✅ Data preparation complete!")
    print("📁 Files created:")
    print("  - data/amazon_reviews_multilingual.csv")
    print("  - data/sample_business_feedback.csv")