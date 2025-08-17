#!/usr/bin/env python3
"""
3D Sentiment & Aspect Analysis Visualizer
Creates interactive 3D visualizations showing sentiment clustering and aspect categorization
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append('src')
sys.path.append('src/models')

class SentimentAspect3DVisualizer:
    """Create stunning 3D visualizations for sentiment and aspect analysis"""
    
    def __init__(self):
        self.df = None
        self.processed_data = None
        self.embeddings_3d = None
        
        # Color schemes
        self.sentiment_colors = {
            'positive': '#2E8B57',    # Sea Green
            'negative': '#DC143C',    # Crimson  
            'neutral': '#4682B4'      # Steel Blue
        }
        
        self.aspect_colors = {
            'product_quality': '#FF6B35',     # Orange-Red
            'user_experience': '#4ECDC4',     # Teal
            'general': '#95A5A6'              # Gray
        }
        
        print("🎨 3D Sentiment & Aspect Visualizer Initialized")
    
    def load_and_process_data(self, csv_file=None):
        """Load data and run ML analysis"""
        
        if csv_file and os.path.exists(csv_file):
            print(f"📊 Loading data from {csv_file}")
            self.df = pd.read_csv(csv_file)
        else:
            print("🔍 Looking for FedEx reviews data...")
            # Find the most recent FedEx CSV file
            data_files = [f for f in os.listdir('data') if f.startswith('fedex_reviews_') and f.endswith('.csv')]
            if data_files:
                latest_file = f"data/{sorted(data_files)[-1]}"
                print(f"📊 Loading latest data: {latest_file}")
                self.df = pd.read_csv(latest_file)
            else:
                print("🎯 Creating sample data for demonstration...")
                self.df = self._create_demo_data()
        
        print(f"✅ Loaded {len(self.df)} reviews")
        
        # If we don't have ML predictions, run them
        if 'predicted_sentiment' not in self.df.columns:
            print("🤖 Running ML analysis...")
            self._run_ml_analysis()
        
        return self.df
    
    def _create_demo_data(self):
        """Create comprehensive demo data for visualization"""
        
        demo_texts = {
            # Positive Product Quality
            'pos_quality': [
                "Excellent tracking accuracy, always shows correct package location",
                "Very reliable delivery service, packages arrive on time consistently", 
                "Outstanding build quality, app never crashes during important tracking",
                "Fast and accurate notifications, performance is top-notch",
                "Robust system that handles multiple package tracking flawlessly"
            ],
            
            # Negative Product Quality  
            'neg_quality': [
                "App crashes frequently when trying to track packages",
                "Tracking information is often wrong or severely delayed",
                "Poor notification system, never tells me when packages arrive",
                "Unreliable performance, constantly losing connection to servers",
                "Buggy software with many errors that affect package tracking"
            ],
            
            # Positive User Experience
            'pos_ux': [
                "Very intuitive interface, easy to navigate and find features",
                "Simple and clean design, everything is where you expect it",
                "User-friendly layout makes package management effortless", 
                "Straightforward to use, even for non-tech-savvy users",
                "Beautiful interface design with smooth navigation experience"
            ],
            
            # Negative User Experience
            'neg_ux': [
                "Confusing interface layout, hard to find basic tracking features",
                "Complicated navigation, buttons and menus are poorly organized",
                "Difficult to understand, interface is cluttered and messy",
                "Unintuitive design, takes forever to figure out how to track",
                "Frustrating user experience, nothing is where it should be"
            ],
            
            # Neutral/Mixed
            'neutral': [
                "App works fine, nothing special but gets the job done",
                "Average experience, some good features and some problems",
                "Decent tracking app, has both strengths and weaknesses",
                "Okay overall, meets basic needs but could be improved",
                "Standard delivery app, functions as expected most of the time"
            ]
        }
        
        # Create DataFrame
        rows = []
        for category, texts in demo_texts.items():
            for i, text in enumerate(texts):
                # Determine sentiment and aspect from category
                if 'pos' in category:
                    sentiment = 'positive'
                elif 'neg' in category:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                if 'quality' in category:
                    aspect = 'product_quality'
                elif 'ux' in category:
                    aspect = 'user_experience'
                else:
                    aspect = 'general'
                
                rating = 5 if sentiment == 'positive' else (2 if sentiment == 'negative' else 3)
                
                rows.append({
                    'text': text,
                    'sentiment': sentiment,
                    'aspect': aspect,
                    'rating': rating,
                    'language_detected': 'en',
                    'country': 'us'
                })
        
        return pd.DataFrame(rows)
    
    def _run_ml_analysis(self):
        """Run ML analysis if predictions don't exist"""
        try:
            from integrated_ml_pipeline import IntegratedMLPipeline
            
            pipeline = IntegratedMLPipeline()
            print("🔮 Analyzing texts with ML pipeline...")
            
            # Analyze all texts
            results = []
            for _, row in self.df.iterrows():
                result = pipeline.analyze_text(row['text'])
                results.append(result)
            
            # Add predictions to dataframe
            for i, result in enumerate(results):
                self.df.loc[i, 'predicted_sentiment'] = result['sentiment']
                self.df.loc[i, 'sentiment_confidence'] = result['sentiment_confidence']
                self.df.loc[i, 'predicted_aspect'] = result['aspect']
                self.df.loc[i, 'aspect_confidence'] = result['aspect_confidence']
            
            print("✅ ML analysis complete!")
            
        except Exception as e:
            print(f"⚠️ ML analysis failed: {e}")
            print("📊 Using existing sentiment/aspect labels")
            
            # Use existing labels if available
            if 'sentiment' in self.df.columns:
                self.df['predicted_sentiment'] = self.df['sentiment']
                self.df['sentiment_confidence'] = 0.85
            if 'aspect' in self.df.columns:
                self.df['predicted_aspect'] = self.df['aspect']
                self.df['aspect_confidence'] = 0.80
    
    def create_text_embeddings(self, sample_size=None):
        """Create 3D embeddings for text visualization"""
        
        print("🧠 Creating text embeddings for 3D visualization...")
        
        # Sample data if too large
        if sample_size and len(self.df) > sample_size:
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"📉 Sampled {sample_size} reviews for visualization")
        
        # Create TF-IDF embeddings
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        # Fit and transform text data
        text_embeddings = vectorizer.fit_transform(self.df['text'])
        
        print(f"🔢 Created embeddings: {text_embeddings.shape}")
        
        # Reduce to 3D using t-SNE for better clustering
        print("🎯 Reducing dimensions to 3D with t-SNE...")
        tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(self.df)-1))
        embeddings_3d = tsne.fit_transform(text_embeddings.toarray())
        
        # Store 3D coordinates
        self.df['x'] = embeddings_3d[:, 0]
        self.df['y'] = embeddings_3d[:, 1] 
        self.df['z'] = embeddings_3d[:, 2]
        
        print("✅ 3D embeddings ready!")
        return embeddings_3d
    
    def create_sentiment_3d_plot(self):
        """Create interactive 3D sentiment visualization"""
        
        print("🎨 Creating 3D Sentiment Cluster Visualization...")
        
        fig = go.Figure()
        
        # Add points for each sentiment
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_data = self.df[self.df['predicted_sentiment'] == sentiment]
            
            if len(sentiment_data) == 0:
                continue
            
            # Create hover text with examples
            hover_text = []
            for _, row in sentiment_data.iterrows():
                text_preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
                confidence = row.get('sentiment_confidence', 0.8)
                hover_text.append(
                    f"<b>Sentiment:</b> {sentiment.title()}<br>"
                    f"<b>Confidence:</b> {confidence:.2f}<br>"
                    f"<b>Text:</b> {text_preview}<br>"
                    f"<b>Language:</b> {row.get('language_detected', 'en')}"
                )
            
            fig.add_trace(go.Scatter3d(
                x=sentiment_data['x'],
                y=sentiment_data['y'],
                z=sentiment_data['z'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.sentiment_colors[sentiment],
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                name=f'{sentiment.title()} ({len(sentiment_data)})',
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': '🎭 3D Sentiment Analysis Clustering<br><sub>Interactive visualization of sentiment patterns in customer feedback</sub>',
                'x': 0.5,
                'font': {'size': 20}
            },
            scene=dict(
                xaxis_title='Semantic Dimension 1',
                yaxis_title='Semantic Dimension 2',
                zaxis_title='Semantic Dimension 3',
                bgcolor='rgba(240,240,240,0.1)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            legend=dict(
                x=0,
                y=1,
                bgcolor='rgba(255,255,255,0.8)'
            ),
            width=1000,
            height=700,
            margin=dict(l=0, r=0, b=0, t=80)
        )
        
        return fig
    
    def create_aspect_3d_plot(self):
        """Create interactive 3D aspect categorization visualization"""
        
        print("🎯 Creating 3D Aspect Classification Visualization...")
        
        fig = go.Figure()
        
        # Add points for each aspect
        for aspect in ['product_quality', 'user_experience', 'general']:
            aspect_data = self.df[self.df['predicted_aspect'] == aspect]
            
            if len(aspect_data) == 0:
                continue
            
            # Create hover text with examples
            hover_text = []
            for _, row in aspect_data.iterrows():
                text_preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
                confidence = row.get('aspect_confidence', 0.8)
                sentiment = row.get('predicted_sentiment', 'unknown')
                
                hover_text.append(
                    f"<b>Aspect:</b> {aspect.replace('_', ' ').title()}<br>"
                    f"<b>Confidence:</b> {confidence:.2f}<br>"
                    f"<b>Sentiment:</b> {sentiment.title()}<br>"
                    f"<b>Text:</b> {text_preview}"
                )
            
            # Map aspect to readable name
            aspect_names = {
                'product_quality': 'Product Quality',
                'user_experience': 'User Experience', 
                'general': 'General'
            }
            
            fig.add_trace(go.Scatter3d(
                x=aspect_data['x'],
                y=aspect_data['y'],
                z=aspect_data['z'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.aspect_colors[aspect],
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                name=f'{aspect_names[aspect]} ({len(aspect_data)})',
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': '🎯 3D Aspect Classification Analysis<br><sub>Product Quality vs User Experience categorization</sub>',
                'x': 0.5,
                'font': {'size': 20}
            },
            scene=dict(
                xaxis_title='Semantic Dimension 1',
                yaxis_title='Semantic Dimension 2', 
                zaxis_title='Semantic Dimension 3',
                bgcolor='rgba(240,240,240,0.1)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            legend=dict(
                x=0,
                y=1,
                bgcolor='rgba(255,255,255,0.8)'
            ),
            width=1000,
            height=700,
            margin=dict(l=0, r=0, b=0, t=80)
        )
        
        return fig
    
    def create_combined_analysis_plot(self):
        """Create combined sentiment + aspect analysis"""
        
        print("🌟 Creating Combined 3D Analysis Visualization...")
        
        fig = go.Figure()
        
        # Create combinations of sentiment and aspect
        combinations = []
        for sentiment in ['positive', 'negative', 'neutral']:
            for aspect in ['product_quality', 'user_experience', 'general']:
                combo_data = self.df[
                    (self.df['predicted_sentiment'] == sentiment) & 
                    (self.df['predicted_aspect'] == aspect)
                ]
                
                if len(combo_data) > 0:
                    combinations.append((sentiment, aspect, combo_data))
        
        # Add traces for each combination
        for sentiment, aspect, data in combinations:
            if len(data) == 0:
                continue
            
            # Create unique color for combination
            base_color = self.sentiment_colors[sentiment]
            
            # Adjust marker style based on aspect
            marker_symbol = 'circle' if aspect == 'product_quality' else (
                'diamond' if aspect == 'user_experience' else 'square'
            )
            
            # Create hover text
            hover_text = []
            for _, row in data.iterrows():
                text_preview = row['text'][:60] + "..." if len(row['text']) > 60 else row['text']
                hover_text.append(
                    f"<b>Sentiment:</b> {sentiment.title()}<br>"
                    f"<b>Aspect:</b> {aspect.replace('_', ' ').title()}<br>"
                    f"<b>Text:</b> {text_preview}"
                )
            
            fig.add_trace(go.Scatter3d(
                x=data['x'],
                y=data['y'],
                z=data['z'],
                mode='markers',
                marker=dict(
                    size=7,
                    color=base_color,
                    symbol=marker_symbol,
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                name=f'{sentiment.title()} + {aspect.replace("_", " ").title()} ({len(data)})',
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': '🌟 Combined Sentiment & Aspect Analysis<br><sub>Multi-dimensional view of customer feedback patterns</sub>',
                'x': 0.5,
                'font': {'size': 20}
            },
            scene=dict(
                xaxis_title='Semantic Dimension 1',
                yaxis_title='Semantic Dimension 2',
                zaxis_title='Semantic Dimension 3',
                bgcolor='rgba(240,240,240,0.1)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            legend=dict(
                x=0,
                y=1,
                bgcolor='rgba(255,255,255,0.8)',
                font=dict(size=10)
            ),
            width=1200,
            height=800,
            margin=dict(l=0, r=0, b=0, t=80)
        )
        
        return fig
    
    def create_analysis_dashboard(self):
        """Create comprehensive analysis dashboard with multiple views"""
        
        print("📊 Creating Analysis Dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Sentiment Distribution', 
                'Aspect Distribution',
                'Confidence Scores',
                'Language Distribution'
            ),
            specs=[[{"type": "pie"}, {"type": "pie"}],
                   [{"type": "box"}, {"type": "bar"}]]
        )
        
        # Sentiment distribution pie chart
        sentiment_counts = self.df['predicted_sentiment'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                marker_colors=[self.sentiment_colors[s] for s in sentiment_counts.index],
                name="Sentiment"
            ),
            row=1, col=1
        )
        
        # Aspect distribution pie chart  
        aspect_counts = self.df['predicted_aspect'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=[a.replace('_', ' ').title() for a in aspect_counts.index],
                values=aspect_counts.values,
                marker_colors=[self.aspect_colors[a] for a in aspect_counts.index],
                name="Aspect"
            ),
            row=1, col=2
        )
        
        # Confidence scores box plot
        if 'sentiment_confidence' in self.df.columns:
            fig.add_trace(
                go.Box(
                    y=self.df['sentiment_confidence'],
                    name="Sentiment Confidence",
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
        
        # Language distribution
        if 'language_detected' in self.df.columns:
            lang_counts = self.df['language_detected'].value_counts().head(5)
            fig.add_trace(
                go.Bar(
                    x=lang_counts.index,
                    y=lang_counts.values,
                    marker_color='lightgreen',
                    name="Language"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="📊 ML Analysis Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def generate_sample_examples(self):
        """Generate text examples for each category"""
        
        print("📝 Generating Sample Examples...")
        
        examples = {}
        
        # Get examples for each sentiment-aspect combination
        for sentiment in ['positive', 'negative', 'neutral']:
            examples[sentiment] = {}
            for aspect in ['product_quality', 'user_experience', 'general']:
                combo_data = self.df[
                    (self.df['predicted_sentiment'] == sentiment) & 
                    (self.df['predicted_aspect'] == aspect)
                ]
                
                if len(combo_data) > 0:
                    # Get top confidence examples
                    if 'sentiment_confidence' in combo_data.columns:
                        sample = combo_data.nlargest(2, 'sentiment_confidence')
                    else:
                        sample = combo_data.head(2)
                    
                    examples[sentiment][aspect] = sample['text'].tolist()
        
        return examples
    
    def save_visualizations(self, output_dir="visualizations"):
        """Save all visualizations to HTML files"""
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        print(f"💾 Saving visualizations to {output_dir}/")
        
        # Create and save each visualization
        visualizations = {
            f"sentiment_3d_{timestamp}.html": self.create_sentiment_3d_plot(),
            f"aspect_3d_{timestamp}.html": self.create_aspect_3d_plot(), 
            f"combined_3d_{timestamp}.html": self.create_combined_analysis_plot(),
            f"dashboard_{timestamp}.html": self.create_analysis_dashboard()
        }
        
        for filename, fig in visualizations.items():
            filepath = os.path.join(output_dir, filename)
            py.plot(fig, filename=filepath, auto_open=False)
            print(f"✅ Saved: {filepath}")
        
        return list(visualizations.keys())
    
    def run_complete_analysis(self, csv_file=None, sample_size=200):
        """Run complete 3D analysis pipeline"""
        
        print("🚀 Starting Complete 3D Analysis Pipeline")
        print("="*60)
        
        # Load and process data
        self.load_and_process_data(csv_file)
        
        # Create embeddings
        self.create_text_embeddings(sample_size)
        
        # Generate examples
        examples = self.generate_sample_examples()
        
        # Create visualizations
        print("\n🎨 Creating Interactive Visualizations...")
        
        sentiment_fig = self.create_sentiment_3d_plot()
        aspect_fig = self.create_aspect_3d_plot()
        combined_fig = self.create_combined_analysis_plot()
        dashboard_fig = self.create_analysis_dashboard()
        
        # Save all visualizations
        saved_files = self.save_visualizations()
        
        # Print summary
        print(f"\n📈 Analysis Summary:")
        print(f"   📊 Analyzed {len(self.df)} reviews")
        print(f"   🎭 Sentiment distribution:")
        for sentiment, count in self.df['predicted_sentiment'].value_counts().items():
            print(f"      {sentiment}: {count} ({count/len(self.df)*100:.1f}%)")
        
        print(f"   🎯 Aspect distribution:")
        for aspect, count in self.df['predicted_aspect'].value_counts().items():
            print(f"      {aspect.replace('_', ' ')}: {count} ({count/len(self.df)*100:.1f}%)")
        
        print(f"\n💾 Saved visualizations:")
        for file in saved_files:
            print(f"      {file}")
        
        print(f"\n🌟 Open any HTML file in your browser to view interactive 3D visualizations!")
        
        return {
            'data': self.df,
            'examples': examples,
            'figures': {
                'sentiment': sentiment_fig,
                'aspect': aspect_fig,
                'combined': combined_fig,
                'dashboard': dashboard_fig
            }
        }

# Main execution
if __name__ == "__main__":
    print("🎨 3D Sentiment & Aspect Analysis Visualizer")
    print("="*60)
    
    # Initialize visualizer
    visualizer = SentimentAspect3DVisualizer()
    
    # Run complete analysis
    results = visualizer.run_complete_analysis(
        csv_file=None,  # Will auto-find FedEx data or create demo data
        sample_size=200  # Limit for better performance
    )
    
    print("\n🎉 3D Analysis Complete!")
    print("\n📋 Next Steps:")
    print("   1. Open the generated HTML files in your browser")
    print("   2. Interact with the 3D plots (rotate, zoom, hover)")
    print("   3. Customize colors/styles in the code if needed")