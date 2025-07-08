from app import db
from datetime import datetime

class SentimentAnalysis(db.Model):
    """Model to store sentiment analysis results"""
    __tablename__ = 'sentiment_analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(50), nullable=False)  # Display sentiment (Alegria, Tristeza, etc.)
    sentiment_key = db.Column(db.String(50), nullable=False)  # Internal key (alegria, tristeza, etc.)
    confidence = db.Column(db.Float, nullable=False)
    probabilities = db.Column(db.JSON, nullable=False)  # Store probability distribution
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    ip_address = db.Column(db.String(45))  # IPv4 or IPv6 address
    user_agent = db.Column(db.String(500))  # Browser/client information
    
    def __repr__(self):
        return f'<SentimentAnalysis {self.id}: {self.sentiment}>'
    
    def to_dict(self):
        """Convert model to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'text': self.text,
            'sentiment': self.sentiment,
            'sentiment_key': self.sentiment_key,
            'confidence': self.confidence,
            'probabilities': self.probabilities,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent
        }

class AnalyticsCache(db.Model):
    """Model to cache analytics data for performance"""
    __tablename__ = 'analytics_cache'
    
    id = db.Column(db.Integer, primary_key=True)
    cache_key = db.Column(db.String(100), unique=True, nullable=False)
    data = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    
    def __repr__(self):
        return f'<AnalyticsCache {self.cache_key}>'
    
    def is_expired(self):
        """Check if cache entry is expired"""
        return datetime.utcnow() > self.expires_at