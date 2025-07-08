import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
try:
    from sentence_transformers import SentenceTransformer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using basic features only.")
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import balance_dataset, create_visualizations

class SentimentModel:
    def __init__(self):
        if TRANSFORMERS_AVAILABLE:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        else:
            self.device = 'cpu'
            self.bert_model = None
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.is_trained = False
        
    def load_and_prepare_data(self, csv_path):
        """Load and prepare dataset for training"""
        print("Loading dataset...")
        df = pd.read_csv(csv_path)
        
        # Clean data
        df = df.dropna(subset=['cleaned_review'])
        df['cleaned_review'] = df['cleaned_review'].astype(str)
        df['sentiments'] = df['sentiments'].str.lower()
        
        # Map sentiments to Portuguese labels for consistency
        sentiment_mapping = {
            'positive': 'alegria',
            'negative': 'tristeza', 
            'neutral': 'surpresa'
        }
        df['sentiments'] = df['sentiments'].map(sentiment_mapping).fillna(df['sentiments'])
        
        # Balance dataset
        df_balanced = balance_dataset(df, 'sentiments')
        
        return df_balanced
    
    def generate_embeddings(self, texts, cache_file="embeddings_bert.npy"):
        """Generate or load cached BERT embeddings"""
        if os.path.exists(cache_file):
            print("Loading embeddings from cache...")
            embeddings = np.load(cache_file)
        else:
            print("Generating BERT embeddings (this may take a while)...")
            embeddings = self.bert_model.encode(
                texts,
                batch_size=128,
                show_progress_bar=True,
                device=self.device,
                convert_to_numpy=True
            )
            np.save(cache_file, embeddings)
            print(f"Embeddings cached to {cache_file}")
        
        return embeddings
    
    def train(self, csv_path="data/amazon_review_comments.csv"):
        """Train the sentiment classification model"""
        # Load and prepare data
        df = self.load_and_prepare_data(csv_path)
        
        # Generate embeddings
        X = self.generate_embeddings(df['cleaned_review'].tolist())
        y = df['sentiments'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Train classifier
        print("Training logistic regression classifier...")
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate model
        y_pred = self.classifier.predict(X_test)
        
        print("\n=== Classification Report ===")
        report = classification_report(y_test, y_pred, output_dict=True)
        print(classification_report(y_test, y_pred))
        
        print("\n=== Confusion Matrix ===")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Create and save visualizations
        create_visualizations(y_test, y_pred, self.classifier.classes_, report, cm)
        
        # Save model
        self.save_model()
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'classes': self.classifier.classes_.tolist()
        }
    
    def save_model(self, model_path="models/sentiment_model.pkl"):
        """Save trained model to disk"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'classifier': self.classifier,
            'classes': self.classifier.classes_,
            'device': self.device,
            'is_trained': self.is_trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path="models/sentiment_model.pkl"):
        """Load trained model from disk"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {model_path}")

def main():
    """Main training script"""
    model = SentimentModel()
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # Train model
    results = model.train()
    
    print("\nTraining completed successfully!")
    print(f"Model classes: {results['classes']}")

if __name__ == "__main__":
    main()
