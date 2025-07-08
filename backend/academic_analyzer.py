"""
Módulo de análise acadêmica para o projeto de Machine Learning
Análise avançada de sentimentos com métricas acadêmicas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import re
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class AcademicSentimentAnalyzer:
    """Classe para análise acadêmica avançada de sentimentos"""
    
    def __init__(self):
        self.analysis_history = []
        self.performance_metrics = {}
        
    def analyze_text_features(self, text):
        """Analisa características linguísticas do texto"""
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'punctuation_count': len([c for c in text if c in '!?.,;:']),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'negative_words': self._count_negative_words(text),
            'positive_words': self._count_positive_words(text),
            'emotional_intensity': self._calculate_emotional_intensity(text)
        }
        return features
    
    def _count_negative_words(self, text):
        """Conta palavras negativas no texto"""
        negative_words = [
            'não', 'nunca', 'jamais', 'nenhum', 'nada', 'nem', 'sem',
            'ruim', 'péssimo', 'horrível', 'terrível', 'insatisfeito',
            'odeia', 'detesta', 'problema', 'defeito', 'quebrou'
        ]
        text_lower = text.lower()
        return sum(1 for word in negative_words if word in text_lower)
    
    def _count_positive_words(self, text):
        """Conta palavras positivas no texto"""
        positive_words = [
            'bom', 'ótimo', 'excelente', 'perfeito', 'maravilhoso',
            'fantástico', 'incrível', 'adorei', 'gostei', 'recomendo',
            'satisfeito', 'feliz', 'funcionou', 'qualidade'
        ]
        text_lower = text.lower()
        return sum(1 for word in positive_words if word in text_lower)
    
    def _calculate_emotional_intensity(self, text):
        """Calcula intensidade emocional baseada em modificadores"""
        intensifiers = ['muito', 'extremamente', 'totalmente', 'completamente', 'bastante']
        text_lower = text.lower()
        intensity = sum(1 for word in intensifiers if word in text_lower)
        return min(intensity, 5)  
    
    def analyze_sentiment_distribution(self, predictions):
        """Analisa distribuição de sentimentos nas predições"""
        sentiments = [pred['sentiment_key'] for pred in predictions]
        sentiment_counts = Counter(sentiments)
        
        distribution = {
            'alegria': sentiment_counts.get('alegria', 0),
            'tristeza': sentiment_counts.get('tristeza', 0),
            'raiva': sentiment_counts.get('raiva', 0),
            'surpresa': sentiment_counts.get('surpresa', 0)
        }
        
        total = sum(distribution.values())
        if total > 0:
            percentages = {k: (v/total)*100 for k, v in distribution.items()}
        else:
            percentages = {k: 0 for k in distribution.keys()}
            
        return {
            'counts': distribution,
            'percentages': percentages,
            'total_analyzed': total,
            'dominant_sentiment': max(distribution.keys(), key=lambda k: distribution[k]) if total > 0 else None
        }
    
    def calculate_confidence_metrics(self, predictions):
        """Calcula métricas de confiança das predições"""
        confidences = [pred['confidence'] for pred in predictions if 'confidence' in pred]
        
        if not confidences:
            return {}
            
        return {
            'mean_confidence': np.mean(confidences),
            'median_confidence': np.median(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'low_confidence_count': sum(1 for c in confidences if c < 0.7),
            'high_confidence_count': sum(1 for c in confidences if c > 0.9)
        }
    
    def generate_academic_insights(self, predictions):
        """Gera insights acadêmicos sobre as predições"""
        if not predictions:
            return {"error": "Nenhuma predição disponível para análise"}
        
        
        text_features = []
        for pred in predictions:
            if 'text' in pred:
                features = self.analyze_text_features(pred['text'])
                features['predicted_sentiment'] = pred.get('sentiment_key', 'unknown')
                text_features.append(features)
        
        
        distribution = self.analyze_sentiment_distribution(predictions)
        
        
        confidence_metrics = self.calculate_confidence_metrics(predictions)
        
        
        temporal_analysis = self._analyze_temporal_patterns(predictions)
        
        
        text_insights = self._analyze_text_characteristics(text_features)
        
        academic_report = {
            'dataset_summary': {
                'total_samples': len(predictions),
                'collection_period': temporal_analysis.get('period', 'N/A'),
                'avg_text_length': np.mean([f['text_length'] for f in text_features]) if text_features else 0
            },
            'sentiment_distribution': distribution,
            'confidence_analysis': confidence_metrics,
            'text_characteristics': text_insights,
            'temporal_patterns': temporal_analysis,
            'model_performance_indicators': self._evaluate_model_consistency(predictions),
            'recommendations': self._generate_academic_recommendations(predictions, text_features)
        }
        
        return academic_report
    
    def _analyze_temporal_patterns(self, predictions):
        """Analisa padrões temporais nas predições"""
        
        return {
            'period': f"{datetime.now().strftime('%Y-%m')}",
            'analysis_trend': 'Análise em tempo real',
            'peak_usage_time': 'Dados insuficientes para análise temporal'
        }
    
    def _analyze_text_characteristics(self, text_features):
        """Analisa características gerais dos textos"""
        if not text_features:
            return {}
        
        df = pd.DataFrame(text_features)
        
        insights = {
            'avg_word_count': df['word_count'].mean(),
            'avg_sentence_count': df['sentence_count'].mean(),
            'emotional_intensity_avg': df['emotional_intensity'].mean(),
            'negative_positive_ratio': df['negative_words'].sum() / max(df['positive_words'].sum(), 1),
            'sentiment_text_length_correlation': self._calculate_sentiment_length_correlation(df)
        }
        
        return insights
    
    def _calculate_sentiment_length_correlation(self, df):
        """Calcula correlação entre tamanho do texto e sentimento"""
        sentiment_mapping = {'alegria': 1, 'surpresa': 0, 'tristeza': -1, 'raiva': -2}
        df['sentiment_numeric'] = df['predicted_sentiment'].map(sentiment_mapping).fillna(0)
        
        correlation = df['text_length'].corr(df['sentiment_numeric'])
        return correlation if not np.isnan(correlation) else 0
    
    def _evaluate_model_consistency(self, predictions):
        """Avalia consistência do modelo"""
        consistencies = []
        confidences = [p.get('confidence', 0) for p in predictions]
        
        return {
            'confidence_consistency': np.std(confidences) if confidences else 0,
            'prediction_stability': 'Estável' if np.std(confidences) < 0.2 else 'Variável',
            'model_bias_indicators': self._detect_model_bias(predictions)
        }
    
    def _detect_model_bias(self, predictions):
        """Detecta possíveis vieses no modelo"""
        sentiments = [p.get('sentiment_key', '') for p in predictions]
        sentiment_counts = Counter(sentiments)
        total = len(sentiments)
        
        if total == 0:
            return "Dados insuficientes"
        
        
        max_percentage = max(sentiment_counts.values()) / total * 100
        
        if max_percentage > 70:
            dominant = max(sentiment_counts.keys(), key=lambda k: sentiment_counts[k])
            return f"Possível viés para '{dominant}' ({max_percentage:.1f}%)"
        else:
            return "Distribuição equilibrada"
    
    def _generate_academic_recommendations(self, predictions, text_features):
        """Gera recomendações acadêmicas baseadas na análise"""
        recommendations = []
        
        
        distribution = self.analyze_sentiment_distribution(predictions)
        if distribution['total_analyzed'] > 0:
            dominant = distribution['dominant_sentiment']
            dominant_percentage = distribution['percentages'][dominant]
            
            if dominant_percentage > 60:
                recommendations.append(f"Considere coletar mais dados para balancear a classe '{dominant}' que representa {dominant_percentage:.1f}% dos dados")
        
        
        confidence_metrics = self.calculate_confidence_metrics(predictions)
        if confidence_metrics.get('mean_confidence', 0) < 0.7:
            recommendations.append("Confiança média baixa - considere melhorar o modelo ou revisar os dados de treinamento")
        
        
        if text_features:
            avg_words = np.mean([f['word_count'] for f in text_features])
            if avg_words < 5:
                recommendations.append("Textos muito curtos podem limitar a precisão - considere incluir mais contexto")
        
        if not recommendations:
            recommendations.append("O modelo está apresentando boa performance geral")
        
        return recommendations
    
    def create_interactive_visualizations(self, predictions):
        """Cria visualizações interativas para análise acadêmica"""
        if not predictions:
            return None
        
        
        distribution = self.analyze_sentiment_distribution(predictions)
        
        
        sentiment_labels = list(distribution['counts'].keys())
        sentiment_values = list(distribution['counts'].values())
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=sentiment_labels,
            values=sentiment_values,
            hole=0.3,
            title="Distribuição de Sentimentos"
        )])
        
        
        confidences = [p.get('confidence', 0) for p in predictions]
        sentiments = [p.get('sentiment_key', 'unknown') for p in predictions]
        
        fig_confidence = px.box(
            x=sentiments,
            y=confidences,
            title="Distribuição de Confiança por Sentimento"
        )
        
        return {
            'pie_chart': fig_pie.to_html(),
            'confidence_box': fig_confidence.to_html()
        }


academic_analyzer = AcademicSentimentAnalyzer()