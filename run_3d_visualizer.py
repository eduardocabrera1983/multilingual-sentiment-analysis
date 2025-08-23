#!/usr/bin/env python3
"""
Advanced 3D Multi-Label Sentiment & Aspect Visualizer
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
import subprocess
warnings.filterwarnings('ignore')

def install_required_packages():
    """Install required packages for 3D visualization"""
    required_packages = [
        'plotly>=5.0.0',
        'scikit-learn>=1.0.0', 
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'umap-learn>=0.5.0'  # For better dimensionality reduction
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
        
        for package in required_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
        
        print("✅ Packages installed successfully!")
        return True

class MultiLabel3DVisualizer:
    """Advanced 3D visualizer for multi-label sentiment and aspect analysis"""
    
    def __init__(self):
        self.df = None
        self.processed_data = None
        self.embeddings_3d = None
        
        # Color schemes for multi-label visualization
        self.sentiment_colors = {
            'positive': '#2E8B57',    # Sea Green
            'negative': '#DC143C',    # Crimson  
            'neutral': '#4682B4'      # Steel Blue
        }
        
        self.aspect_colors = {
            'user_experience': '#FF6B35',      # Orange-Red (Priority)
            'performance': '#E74C3C',          # Red (Critical)
            'tracking_accuracy': '#3498DB',    # Blue (Core)
            'delivery_issues': '#F39C12',      # Orange (Important)
            'interface_design': '#9B59B6',     # Purple (Design)
            'general_satisfaction': '#95A5A6'  # Gray (General)
        }
        
        self.classification_colors = {
            'single_aspect': '#2ECC71',     # Green
            'dual_aspect': '#F1C40F',       # Yellow
            'mixed_concerns': '#E74C3C'     # Red
        }
        
        self.priority_colors = {
            'HIGH': '#E74C3C',      # Red
            'MEDIUM': '#F39C12',    # Orange
            'LOW': '#2ECC71'        # Green
        }
        
        print("🎨 Advanced 3D Multi-Label Visualizer Initialized")
    
    def load_and_process_data(self, csv_file=None):
        """Load data and run ML analysis"""
        
        # Add paths for ML pipeline
        sys.path.append('src')
        sys.path.append('src/models')
        
        if csv_file and os.path.exists(csv_file):
            print(f"📊 Loading data from {csv_file}")
            self.df = pd.read_csv(csv_file)
        else:
            print("🔍 Looking for FedEx reviews data...")
            data_files = [f for f in os.listdir('data') if f.startswith('fedex_reviews_') and f.endswith('.csv')]
            if data_files:
                latest_file = f"data/{sorted(data_files)[-1]}"
                print(f"📊 Loading latest data: {latest_file}")
                self.df = pd.read_csv(latest_file)
            else:
                print("🎯 Creating comprehensive demo data for presentation...")
                self.df = self._create_presentation_demo_data()
        
        print(f"✅ Loaded {len(self.df)} reviews")
        
        # Run ML analysis if predictions don't exist
        if 'predicted_primary_aspect' not in self.df.columns:
            print("🤖 Running Multi-Label ML Analysis...")
            self._run_multilabel_analysis()
        
        return self.df
    
    def _create_presentation_demo_data(self):
        """Create comprehensive demo data showcasing all multi-label features"""
        
        demo_data = {
            # Single Aspect Examples
            'single_ux': [
                "Interface is impossible to use, terrible navigation and confusing layout",
                "Very intuitive interface, easy to navigate and find features",
                "Confusing interface layout, hard to find basic tracking features"
            ],
            'single_performance': [
                "App crashes constantly when trying to track packages",
                "Fast and smooth performance, works perfectly every time",
                "App freezes frequently, completely broken functionality"
            ],
            'single_tracking': [
                "Tracking is very accurate, always shows correct package location",
                "Tracking information is completely wrong, shows incorrect status",
                "Real-time updates are perfect, always know where my package is"
            ],
            
            # Dual Aspect Examples
            'dual_ux_tracking': [
                "Love the tracking accuracy but the interface is confusing",
                "Great tracking features but navigation is difficult",
                "Accurate tracking but interface needs improvement"
            ],
            'dual_performance_ux': [
                "App crashes frequently and the interface is terrible",
                "Fast app but interface design is confusing",
                "Performance is smooth but navigation is unintuitive"
            ],
            'dual_tracking_delivery': [
                "Tracking works great but deliveries are always late",
                "Perfect tracking information but delivery service is poor",
                "Accurate status updates but packages never arrive on time"
            ],
            
            # Mixed Concerns Examples
            'mixed_complex': [
                "App crashes, interface is terrible, and deliveries are always late",
                "Slow performance, confusing navigation, and wrong tracking information",
                "Buggy app, poor design, and unreliable delivery service"
            ],
            
            # Positive Mixed Examples
            'positive_mixed': [
                "Great tracking, beautiful interface, and fast delivery",
                "Excellent performance, intuitive design, and reliable service",
                "Perfect app with accurate tracking and outstanding delivery"
            ],
            
            # Business Priority Examples
            'critical_issues': [
                "This app is completely unusable, crashes every time I try to track",
                "Interface is impossible to navigate, worst app experience ever",
                "App doesn't work at all, complete disaster and waste of time"
            ]
        }
        
        # Create DataFrame
        rows = []
        for category, texts in demo_data.items():
            for text in texts:
                # Determine expected sentiment
                if 'positive' in category or any(word in text.lower() for word in ['great', 'perfect', 'excellent', 'love', 'beautiful']):
                    sentiment = 'positive'
                    rating = 5
                elif any(word in text.lower() for word in ['terrible', 'worst', 'crash', 'broken', 'impossible', 'disaster']):
                    sentiment = 'negative'
                    rating = 1
                else:
                    sentiment = 'neutral'
                    rating = 3
                
                # Determine expected classification type
                if 'single' in category:
                    expected_type = 'single_aspect'
                elif 'dual' in category:
                    expected_type = 'dual_aspect'
                else:
                    expected_type = 'mixed_concerns'
                
                rows.append({
                    'text': text,
                    'rating': rating,
                    'sentiment': sentiment,
                    'category': category,
                    'expected_type': expected_type,
                    'language_detected': 'en',
                    'country': 'us'
                })
        
        return pd.DataFrame(rows)
    
    def _run_multilabel_analysis(self):
        """Run multi-label analysis using the integrated pipeline"""
        try:
            from integrated_ml_pipeline import IntegratedMLPipeline
            
            pipeline = IntegratedMLPipeline()
            print("🔮 Analyzing texts with Multi-Label ML Pipeline...")
            
            # Analyze all texts
            results = []
            for _, row in self.df.iterrows():
                result = pipeline.analyze_text(row['text'])
                results.append(result)
            
            # Add predictions to dataframe
            for i, result in enumerate(results):
                self.df.loc[i, 'predicted_sentiment'] = result['sentiment']
                self.df.loc[i, 'sentiment_confidence'] = result['sentiment_confidence']
                self.df.loc[i, 'predicted_primary_aspect'] = result['primary_aspect']
                self.df.loc[i, 'predicted_secondary_aspects'] = str(result['secondary_aspects'])
                self.df.loc[i, 'predicted_classification_type'] = result['classification_type']
                self.df.loc[i, 'predicted_priority_level'] = result['priority_level']
                self.df.loc[i, 'predicted_severity_level'] = result['severity_level']
                self.df.loc[i, 'requires_immediate_action'] = result['requires_immediate_action']
            
            print("✅ Multi-Label ML analysis complete!")
            
        except Exception as e:
            print(f"⚠️ ML analysis failed: {e}")
            print("📊 Using demo classifications for visualization")
            self._add_demo_classifications()
    
    def _add_demo_classifications(self):
        """Add demo classifications if ML pipeline fails"""
        for i, row in self.df.iterrows():
            # Use simple keyword-based classification for demo
            text = row['text'].lower()
            
            # Primary aspect
            if any(word in text for word in ['interface', 'navigation', 'design', 'layout', 'user']):
                primary = 'user_experience'
            elif any(word in text for word in ['crash', 'bug', 'freeze', 'performance', 'slow']):
                primary = 'performance'
            elif any(word in text for word in ['tracking', 'track', 'location', 'status']):
                primary = 'tracking_accuracy'
            elif any(word in text for word in ['delivery', 'deliver', 'package', 'shipping']):
                primary = 'delivery_issues'
            else:
                primary = 'general_satisfaction'
            
            # Simple secondary aspect detection
            aspects_found = []
            if any(word in text for word in ['interface', 'navigation', 'design']):
                aspects_found.append('user_experience')
            if any(word in text for word in ['crash', 'bug', 'freeze', 'performance']):
                aspects_found.append('performance')
            if any(word in text for word in ['tracking', 'track', 'location']):
                aspects_found.append('tracking_accuracy')
            if any(word in text for word in ['delivery', 'deliver', 'package']):
                aspects_found.append('delivery_issues')
            
            # Remove primary from secondary
            secondary_aspects = [a for a in aspects_found if a != primary]
            
            # Classification type
            if len(secondary_aspects) == 0:
                class_type = 'single_aspect'
            elif len(secondary_aspects) == 1:
                class_type = 'dual_aspect'
            else:
                class_type = 'mixed_concerns'
            
            # Priority
            if primary in ['user_experience', 'performance']:
                priority = 'HIGH'
            elif any(word in text for word in ['terrible', 'worst', 'impossible', 'disaster']):
                priority = 'HIGH'
            else:
                priority = 'MEDIUM'
            
            # Update dataframe
            self.df.loc[i, 'predicted_primary_aspect'] = primary
            self.df.loc[i, 'predicted_secondary_aspects'] = str(secondary_aspects)
            self.df.loc[i, 'predicted_classification_type'] = class_type
            self.df.loc[i, 'predicted_priority_level'] = priority
            self.df.loc[i, 'predicted_severity_level'] = 'HIGH' if priority == 'HIGH' else 'MODERATE'
    
    def create_text_embeddings(self, sample_size=None):
        """Create 3D embeddings for text visualization using advanced techniques"""
        
        print("🧠 Creating 3D text embeddings for visualization...")
        
        # Sample data if too large
        if sample_size and len(self.df) > sample_size:
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"📉 Sampled {sample_size} reviews for optimal visualization")
        
        # Create enhanced TF-IDF embeddings
        vectorizer = TfidfVectorizer(
            max_features=2000,  # Increased for better representation
            ngram_range=(1, 3), # Include trigrams for better context
            stop_words='english',
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        
        # Fit and transform text data
        text_embeddings = vectorizer.fit_transform(self.df['text'])
        
        print(f"🔢 Created embeddings: {text_embeddings.shape}")
        
        # Use UMAP for better 3D projection (if available)
        try:
            import umap
            print("🎯 Reducing dimensions to 3D with UMAP...")
            reducer = umap.UMAP(n_components=3, random_state=42, min_dist=0.1, n_neighbors=15)
            embeddings_3d = reducer.fit_transform(text_embeddings.toarray())
            print("✅ UMAP 3D embeddings created!")
        except ImportError:
            print("🎯 Reducing dimensions to 3D with t-SNE...")
            tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(self.df)-1))
            embeddings_3d = tsne.fit_transform(text_embeddings.toarray())
            print("✅ t-SNE 3D embeddings created!")
        
        # Store 3D coordinates
        self.df['x'] = embeddings_3d[:, 0]
        self.df['y'] = embeddings_3d[:, 1] 
        self.df['z'] = embeddings_3d[:, 2]
        
        return embeddings_3d
    
    def create_sentiment_3d_plot(self):
        """Create interactive 3D sentiment visualization"""
        
        print("🎭 Creating 3D Sentiment Cluster Visualization...")
        
        fig = go.Figure()
        
        # Add points for each sentiment with enhanced styling
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_data = self.df[self.df.get('predicted_sentiment', self.df.get('sentiment', 'neutral')) == sentiment]
            
            if len(sentiment_data) == 0:
                continue
            
            # Create detailed hover text
            hover_text = []
            for _, row in sentiment_data.iterrows():
                text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
                primary = row.get('predicted_primary_aspect', 'unknown')
                secondary = row.get('predicted_secondary_aspects', '[]')
                class_type = row.get('predicted_classification_type', 'unknown')
                
                hover_text.append(
                    f"<b>Sentiment:</b> {sentiment.title()}<br>"
                    f"<b>Primary Aspect:</b> {primary.replace('_', ' ').title()}<br>"
                    f"<b>Secondary:</b> {secondary}<br>"
                    f"<b>Type:</b> {class_type.replace('_', ' ').title()}<br>"
                    f"<b>Text:</b> {text_preview}<br>"
                )
            
            fig.add_trace(go.Scatter3d(
                x=sentiment_data['x'],
                y=sentiment_data['y'],
                z=sentiment_data['z'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.sentiment_colors[sentiment],
                    opacity=0.8,
                    line=dict(width=2, color='white'),
                    symbol='circle'
                ),
                name=f'{sentiment.title()} Sentiment ({len(sentiment_data)})',
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Update layout with professional styling
        fig.update_layout(
            title={
                'text': '🎭 3D Multi-Label Sentiment Analysis<br><sub>Interactive clustering of customer feedback sentiment</sub>',
                'x': 0.5,
                'font': {'size': 24, 'color': '#2C3E50'}
            },
            scene=dict(
                xaxis_title='Semantic Dimension 1',
                yaxis_title='Semantic Dimension 2',
                zaxis_title='Semantic Dimension 3',
                bgcolor='rgba(240,248,255,0.1)',
                xaxis=dict(gridcolor='rgba(100,100,100,0.3)'),
                yaxis=dict(gridcolor='rgba(100,100,100,0.3)'),
                zaxis=dict(gridcolor='rgba(100,100,100,0.3)'),
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.8))
            ),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            ),
            width=1200,
            height=800,
            margin=dict(l=0, r=0, b=0, t=100),
            font=dict(family='Arial, sans-serif')
        )
        
        return fig
    
    def create_aspect_3d_plot(self):
        """Create interactive 3D primary aspect visualization"""
        
        print("🎯 Creating 3D Primary Aspect Classification...")
        
        fig = go.Figure()
        
        # Get unique primary aspects
        primary_col = 'predicted_primary_aspect' if 'predicted_primary_aspect' in self.df.columns else 'primary_aspect'
        
        for aspect in self.aspect_colors.keys():
            aspect_data = self.df[self.df.get(primary_col, 'general_satisfaction') == aspect]
            
            if len(aspect_data) == 0:
                continue
            
            # Create detailed hover information
            hover_text = []
            for _, row in aspect_data.iterrows():
                text_preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
                sentiment = row.get('predicted_sentiment', row.get('sentiment', 'unknown'))
                secondary = row.get('predicted_secondary_aspects', '[]')
                priority = row.get('predicted_priority_level', 'MEDIUM')
                
                hover_text.append(
                    f"<b>Primary Aspect:</b> {aspect.replace('_', ' ').title()}<br>"
                    f"<b>Secondary:</b> {secondary}<br>"
                    f"<b>Sentiment:</b> {sentiment.title()}<br>"
                    f"<b>Priority:</b> {priority}<br>"
                    f"<b>Text:</b> {text_preview}"
                )
            
            fig.add_trace(go.Scatter3d(
                x=aspect_data['x'],
                y=aspect_data['y'],
                z=aspect_data['z'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.aspect_colors[aspect],
                    opacity=0.8,
                    line=dict(width=2, color='white'),
                    symbol='diamond' if aspect == 'user_experience' else 'circle'
                ),
                name=f'{aspect.replace("_", " ").title()} ({len(aspect_data)})',
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': '🎯 3D Multi-Label Aspect Classification<br><sub>Primary aspect distribution with business prioritization</sub>',
                'x': 0.5,
                'font': {'size': 24, 'color': '#2C3E50'}
            },
            scene=dict(
                xaxis_title='Semantic Dimension 1',
                yaxis_title='Semantic Dimension 2',
                zaxis_title='Semantic Dimension 3',
                bgcolor='rgba(248,249,250,0.1)',
                xaxis=dict(gridcolor='rgba(100,100,100,0.3)'),
                yaxis=dict(gridcolor='rgba(100,100,100,0.3)'),
                zaxis=dict(gridcolor='rgba(100,100,100,0.3)'),
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.8))
            ),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            ),
            width=1200,
            height=800,
            margin=dict(l=0, r=0, b=0, t=100),
            font=dict(family='Arial, sans-serif')
        )
        
        return fig
    
    def create_classification_type_3d_plot(self):
        """Create 3D visualization of multi-label classification types"""
        
        print("🌟 Creating 3D Multi-Label Classification Types...")
        
        fig = go.Figure()
        
        # Get classification type column
        class_col = 'predicted_classification_type' if 'predicted_classification_type' in self.df.columns else 'classification_type'
        
        for class_type in ['single_aspect', 'dual_aspect', 'mixed_concerns']:
            type_data = self.df[self.df.get(class_col, 'single_aspect') == class_type]
            
            if len(type_data) == 0:
                continue
            
            # Different marker shapes for each type
            marker_shapes = {
                'single_aspect': 'circle',
                'dual_aspect': 'diamond',
                'mixed_concerns': 'square'
            }
            
            # Create hover text
            hover_text = []
            for _, row in type_data.iterrows():
                text_preview = row['text'][:70] + "..." if len(row['text']) > 70 else row['text']
                primary = row.get('predicted_primary_aspect', 'unknown')
                secondary = row.get('predicted_secondary_aspects', '[]')
                sentiment = row.get('predicted_sentiment', 'unknown')
                priority = row.get('predicted_priority_level', 'MEDIUM')
                
                hover_text.append(
                    f"<b>Type:</b> {class_type.replace('_', ' ').title()}<br>"
                    f"<b>Primary:</b> {primary.replace('_', ' ').title()}<br>"
                    f"<b>Secondary:</b> {secondary}<br>"
                    f"<b>Sentiment:</b> {sentiment.title()}<br>"
                    f"<b>Priority:</b> {priority}<br>"
                    f"<b>Text:</b> {text_preview}"
                )
            
            fig.add_trace(go.Scatter3d(
                x=type_data['x'],
                y=type_data['y'],
                z=type_data['z'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=self.classification_colors[class_type],
                    opacity=0.8,
                    line=dict(width=2, color='white'),
                    symbol=marker_shapes[class_type]
                ),
                name=f'{class_type.replace("_", " ").title()} ({len(type_data)})',
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': '🌟 3D Multi-Label Classification Types<br><sub>Advanced classification: Single → Dual → Mixed Concerns</sub>',
                'x': 0.5,
                'font': {'size': 24, 'color': '#2C3E50'}
            },
            scene=dict(
                xaxis_title='Semantic Dimension 1',
                yaxis_title='Semantic Dimension 2',
                zaxis_title='Semantic Dimension 3',
                bgcolor='rgba(250,250,250,0.1)',
                xaxis=dict(gridcolor='rgba(100,100,100,0.3)'),
                yaxis=dict(gridcolor='rgba(100,100,100,0.3)'),
                zaxis=dict(gridcolor='rgba(100,100,100,0.3)'),
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.8))
            ),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            ),
            width=1200,
            height=800,
            margin=dict(l=0, r=0, b=0, t=100),
            font=dict(family='Arial, sans-serif')
        )
        
        return fig
    
    def create_business_priority_3d_plot(self):
        """Create 3D visualization of business priority levels"""
        
        print("💼 Creating 3D Business Priority Visualization...")
        
        fig = go.Figure()
        
        priority_col = 'predicted_priority_level' if 'predicted_priority_level' in self.df.columns else 'priority_level'
        
        for priority in ['HIGH', 'MEDIUM', 'LOW']:
            priority_data = self.df[self.df.get(priority_col, 'MEDIUM') == priority]
            
            if len(priority_data) == 0:
                continue
            
            # Marker size based on priority
            marker_sizes = {'HIGH': 14, 'MEDIUM': 10, 'LOW': 8}
            
            hover_text = []
            for _, row in priority_data.iterrows():
                text_preview = row['text'][:60] + "..." if len(row['text']) > 60 else row['text']
                primary = row.get('predicted_primary_aspect', 'unknown')
                class_type = row.get('predicted_classification_type', 'unknown')
                severity = row.get('predicted_severity_level', 'MODERATE')
                
                hover_text.append(
                    f"<b>Priority:</b> {priority}<br>"
                    f"<b>Severity:</b> {severity}<br>"
                    f"<b>Primary Aspect:</b> {primary.replace('_', ' ').title()}<br>"
                    f"<b>Type:</b> {class_type.replace('_', ' ').title()}<br>"
                    f"<b>Text:</b> {text_preview}"
                )
            
            fig.add_trace(go.Scatter3d(
                x=priority_data['x'],
                y=priority_data['y'],
                z=priority_data['z'],
                mode='markers',
                marker=dict(
                    size=marker_sizes[priority],
                    color=self.priority_colors[priority],
                    opacity=0.8,
                    line=dict(width=2, color='white'),
                    symbol='circle'
                ),
                name=f'{priority} Priority ({len(priority_data)})',
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title={
                'text': '💼 3D Business Priority Analysis<br><sub>Automated priority assignment for customer feedback triage</sub>',
                'x': 0.5,
                'font': {'size': 24, 'color': '#2C3E50'}
            },
            scene=dict(
                xaxis_title='Semantic Dimension 1',
                yaxis_title='Semantic Dimension 2',
                zaxis_title='Semantic Dimension 3',
                bgcolor='rgba(245,245,245,0.1)',
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.8))
            ),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            ),
            width=1200,
            height=800,
            margin=dict(l=0, r=0, b=0, t=100)
        )
        
        return fig
    
    def create_comprehensive_dashboard(self):
        """Create comprehensive 2D dashboard with multiple metrics"""
        
        print("📊 Creating Comprehensive Analysis Dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Classification Type Distribution',
                'Primary Aspect Distribution', 
                'Priority Level Distribution',
                'Sentiment vs Priority Heatmap',
                'Business Metrics Summary',
                'Top Aspect Combinations'
            ),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "indicator"}, {"type": "bar"}]]
        )
        
        # 1. Classification Type Pie Chart
        class_col = 'predicted_classification_type' if 'predicted_classification_type' in self.df.columns else 'classification_type'
        class_counts = self.df.get(class_col, pd.Series(['single_aspect'] * len(self.df))).value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=[c.replace('_', ' ').title() for c in class_counts.index],
                values=class_counts.values,
                marker_colors=['#2ECC71', '#F1C40F', '#E74C3C'],
                name="Classification Types"
            ),
            row=1, col=1
        )
        
        # 2. Primary Aspect Bar Chart
        primary_col = 'predicted_primary_aspect' if 'predicted_primary_aspect' in self.df.columns else 'primary_aspect'
        aspect_counts = self.df.get(primary_col, pd.Series(['general_satisfaction'] * len(self.df))).value_counts()
        
        fig.add_trace(
            go.Bar(
                x=[a.replace('_', ' ').title() for a in aspect_counts.index],
                y=aspect_counts.values,
                marker_color='#3498DB',
                name="Primary Aspects"
            ),
            row=1, col=2
        )
        
        # 3. Priority Level Distribution
        priority_col = 'predicted_priority_level' if 'predicted_priority_level' in self.df.columns else 'priority_level'
        priority_counts = self.df.get(priority_col, pd.Series(['MEDIUM'] * len(self.df))).value_counts()
        
        fig.add_trace(
            go.Bar(
                x=priority_counts.index,
                y=priority_counts.values,
                marker_color=['#E74C3C', '#F39C12', '#2ECC71'],
                name="Priority Levels"
            ),
            row=2, col=1
        )
        
        # 4. Business Metrics Indicator
        total_reviews = len(self.df)
        high_priority = len(self.df[self.df.get(priority_col, 'MEDIUM') == 'HIGH'])
        
        fig.add_trace(
            go.Indicator(
                mode = "number+gauge+delta",
                value = (high_priority / total_reviews) * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {"text": "High Priority %"},
                gauge = {'axis': {'range': [None, 100]},
                        'bar': {'color': "#E74C3C"},
                        'steps': [{'range': [0, 50], 'color': "#2ECC71"},
                                 {'range': [50, 75], 'color': "#F39C12"}],
                        'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 75}}
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title_text="📊 Multi-Label Analysis Dashboard",
            height=1000,
            showlegend=False,
            font=dict(family='Arial, sans-serif')
        )
        
        return fig
    
    def save_all_visualizations(self, output_dir="3d_visualizations"):
        """Save all visualizations to HTML files"""
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        print(f"💾 Saving 3D visualizations to {output_dir}/")
        
        # Create all visualizations
        visualizations = {
            f"sentiment_3d_{timestamp}.html": self.create_sentiment_3d_plot(),
            f"aspects_3d_{timestamp}.html": self.create_aspect_3d_plot(),
            f"classification_types_3d_{timestamp}.html": self.create_classification_type_3d_plot(),
            f"business_priority_3d_{timestamp}.html": self.create_business_priority_3d_plot(),
            f"dashboard_{timestamp}.html": self.create_comprehensive_dashboard()
        }
        
        saved_files = []
        for filename, fig in visualizations.items():
            filepath = os.path.join(output_dir, filename)
            py.plot(fig, filename=filepath, auto_open=False)
            print(f"✅ Saved: {filepath}")
            saved_files.append(filename)
        
        return saved_files
    
    def run_complete_3d_analysis(self, csv_file=None, sample_size=200):
        """Run complete 3D visualization pipeline"""
        
        print("🚀 Starting Complete 3D Multi-Label Analysis Pipeline")
        print("="*70)
        
        # Load and process data
        self.load_and_process_data(csv_file)
        
        # Create embeddings
        self.create_text_embeddings(sample_size)
        
        # Generate summary statistics
        print(f"\n📈 Multi-Label Analysis Summary:")
        print(f"   📊 Total Reviews Analyzed: {len(self.df)}")
        
        # Classification type summary
        class_col = 'predicted_classification_type' if 'predicted_classification_type' in self.df.columns else 'classification_type'
        if class_col in self.df.columns:
            print(f"   🎯 Classification Types:")
            for class_type, count in self.df[class_col].value_counts().items():
                print(f"      {class_type.replace('_', ' ').title()}: {count} ({count/len(self.df)*100:.1f}%)")
        
        # Priority summary
        priority_col = 'predicted_priority_level' if 'predicted_priority_level' in self.df.columns else 'priority_level'
        if priority_col in self.df.columns:
            print(f"   💼 Priority Distribution:")
            for priority, count in self.df[priority_col].value_counts().items():
                print(f"      {priority}: {count} ({count/len(self.df)*100:.1f}%)")
        
        # Save all visualizations
        saved_files = self.save_all_visualizations()
        
        print(f"\n💾 Generated 3D Visualizations:")
        for file in saved_files:
            print(f"      {file}")
        
        print(f"\n🌟 3D Analysis Complete!")
        print(f"📱 Open the HTML files in your browser to interact with 3D visualizations")
        
        return {
            'data': self.df,
            'saved_files': saved_files,
            'summary': {
                'total_reviews': len(self.df),
                'classification_types': dict(self.df.get(class_col, pd.Series()).value_counts()) if class_col in self.df.columns else {},
                'priority_levels': dict(self.df.get(priority_col, pd.Series()).value_counts()) if priority_col in self.df.columns else {}
            }
        }

def main():
    """Main execution function"""
    
    print("🚀 Advanced 3D Multi-Label Sentiment Analysis Visualizer")
    print("="*70)
    
    # Install required packages
    if not install_required_packages():
        print("❌ Package installation failed. Please install manually:")
        print("   pip install plotly scikit-learn pandas numpy")
        return
    
    # Initialize visualizer
    visualizer = MultiLabel3DVisualizer()
    
    # Run complete analysis
    results = visualizer.run_complete_3d_analysis(
        csv_file=None,  # Will auto-find FedEx data or create demo data
        sample_size=200  # Optimal for interactive 3D performance
    )
    
    print("\n🎉 3D Multi-Label Visualization System Ready!")
    print("   1. Open the generated HTML files in your browser")
    print("   2. Demonstrate the interactive 3D capabilities") 
    print("   3. Show multi-label classification in 3D space")
    print("   4. Highlight business intelligence insights")
    print("   5. Explain the advanced ML concepts visualized")
    
    print(f"\n🏆 This showcases sophisticated data science skills:")
    print(f"   • Multi-label machine learning classification")
    print(f"   • 3D dimensionality reduction and visualization")
    print(f"   • Interactive data storytelling")
    print(f"   • Business intelligence automation")
    print(f"   • Production-ready ML pipeline integration")

if __name__ == "__main__":
    main()