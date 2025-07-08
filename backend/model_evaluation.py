"""
Módulo de avaliação de modelos para análise de sentimentos
Desenvolvido para projeto acadêmico de Machine Learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import json
import os
from datetime import datetime

class ModelEvaluator:
    """Classe para avaliação completa de modelos de ML"""
    
    def __init__(self, model_name="SentimentModel"):
        self.model_name = model_name
        self.evaluation_results = {}
        self.label_encoder = LabelEncoder()
        
    def evaluate_model(self, model, X_test, y_test, class_names=None):
        """
        Avaliação completa do modelo com múltiplas métricas
        """
        if class_names is None:
            class_names = ['alegria', 'tristeza', 'raiva', 'surpresa']
            
        # Fazer predições
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Tentar obter probabilidades se disponível
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Calcular métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        
        # Relatório de classificação
        class_report = classification_report(y_test, y_pred, 
                                           target_names=class_names, 
                                           output_dict=True, 
                                           zero_division=0)
        
        self.evaluation_results = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'class_names': class_names
        }
        
        return self.evaluation_results
    
    def cross_validate_model(self, model, X, y, cv_folds=5):
        """
        Validação cruzada do modelo
        """
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Scores para diferentes métricas
        accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        precision_scores = cross_val_score(model, X, y, cv=skf, scoring='precision_weighted')
        recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall_weighted')
        f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
        
        cv_results = {
            'cv_folds': cv_folds,
            'accuracy': {
                'mean': float(np.mean(accuracy_scores)),
                'std': float(np.std(accuracy_scores)),
                'scores': accuracy_scores.tolist()
            },
            'precision': {
                'mean': float(np.mean(precision_scores)),
                'std': float(np.std(precision_scores)),
                'scores': precision_scores.tolist()
            },
            'recall': {
                'mean': float(np.mean(recall_scores)),
                'std': float(np.std(recall_scores)),
                'scores': recall_scores.tolist()
            },
            'f1_score': {
                'mean': float(np.mean(f1_scores)),
                'std': float(np.std(f1_scores)),
                'scores': f1_scores.tolist()
            }
        }
        
        self.evaluation_results['cross_validation'] = cv_results
        return cv_results
    
    def plot_confusion_matrix(self, save_path="static/confusion_matrix.png"):
        """
        Gerar e salvar matriz de confusão
        """
        if 'confusion_matrix' not in self.evaluation_results:
            return None
            
        cm = np.array(self.evaluation_results['confusion_matrix'])
        class_names = self.evaluation_results['class_names']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Matriz de Confusão - {self.model_name}')
        plt.xlabel('Predição')
        plt.ylabel('Real')
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_metrics_comparison(self, save_path="static/metrics_comparison.png"):
        """
        Gráfico de barras comparando métricas
        """
        if 'metrics' not in self.evaluation_results:
            return None
            
        metrics = self.evaluation_results['metrics']
        
        plt.figure(figsize=(10, 6))
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        colors = ['#28a745', '#007bff', '#ffc107', '#dc3545']
        bars = plt.bar(metric_names, metric_values, color=colors)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.title(f'Métricas de Avaliação - {self.model_name}')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_cross_validation_results(self, save_path="static/cross_validation.png"):
        """
        Gráfico dos resultados de validação cruzada
        """
        if 'cross_validation' not in self.evaluation_results:
            return None
            
        cv_results = self.evaluation_results['cross_validation']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        colors = ['#28a745', '#007bff', '#ffc107', '#dc3545']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            scores = cv_results[metric]['scores']
            mean_score = cv_results[metric]['mean']
            std_score = cv_results[metric]['std']
            
            # Box plot dos scores
            ax.boxplot(scores, patch_artist=True, 
                      boxprops=dict(facecolor=colors[i], alpha=0.7))
            
            # Linha da média
            ax.axhline(y=mean_score, color='red', linestyle='--', 
                      label=f'Média: {mean_score:.3f} ± {std_score:.3f}')
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.suptitle(f'Validação Cruzada ({cv_results["cv_folds"]} folds) - {self.model_name}')
        plt.tight_layout()
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_academic_report(self, save_path="static/academic_report.json"):
        """
        Gerar relatório acadêmico completo em JSON
        """
        if not self.evaluation_results:
            return None
        
        # Adicionar análise interpretativa
        metrics = self.evaluation_results['metrics']
        
        # Interpretação das métricas
        interpretation = {
            'model_performance': self._interpret_performance(metrics['accuracy']),
            'precision_analysis': self._interpret_metric(metrics['precision'], 'precision'),
            'recall_analysis': self._interpret_metric(metrics['recall'], 'recall'),
            'f1_analysis': self._interpret_metric(metrics['f1_score'], 'f1'),
            'recommendations': self._generate_recommendations()
        }
        
        academic_report = {
            'project_info': {
                'title': 'Análise de Sentimentos em Português usando Machine Learning',
                'objective': 'Classificar sentimentos em texto português usando técnicas de ML/DL',
                'model_type': 'Classificação Multiclasse',
                'classes': self.evaluation_results['class_names'],
                'evaluation_date': self.evaluation_results['timestamp']
            },
            'methodology': {
                'preprocessing': 'Limpeza de texto, tokenização, embeddings',
                'feature_extraction': 'BERT embeddings (all-MiniLM-L6-v2)',
                'algorithm': 'Logistic Regression',
                'validation_method': 'Stratified K-Fold Cross Validation'
            },
            'results': self.evaluation_results,
            'interpretation': interpretation
        }
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(academic_report, f, indent=2, ensure_ascii=False)
        
        return academic_report
    
    def _interpret_performance(self, accuracy):
        """Interpretar performance do modelo"""
        if accuracy >= 0.9:
            return "Excelente: Modelo apresenta alta precisão"
        elif accuracy >= 0.8:
            return "Boa: Performance satisfatória para aplicação prática"
        elif accuracy >= 0.7:
            return "Moderada: Pode necessitar melhorias"
        else:
            return "Baixa: Requer otimização significativa"
    
    def _interpret_metric(self, value, metric_name):
        """Interpretar métrica específica"""
        quality = "alta" if value >= 0.8 else "moderada" if value >= 0.7 else "baixa"
        return f"{metric_name.title()} {quality}: {value:.3f}"
    
    def _generate_recommendations(self):
        """Gerar recomendações baseadas nos resultados"""
        recommendations = []
        
        metrics = self.evaluation_results['metrics']
        
        if metrics['accuracy'] < 0.8:
            recommendations.append("Considerar mais dados de treinamento")
            recommendations.append("Experimentar diferentes algoritmos")
        
        if metrics['precision'] < metrics['recall']:
            recommendations.append("Focar na redução de falsos positivos")
        elif metrics['recall'] < metrics['precision']:
            recommendations.append("Focar na redução de falsos negativos")
        
        if not recommendations:
            recommendations.append("Modelo apresenta performance satisfatória")
            recommendations.append("Considerar validação com dados externos")
        
        return recommendations

def evaluate_sentiment_model():
    """
    Função principal para avaliar o modelo de sentimentos
    """
    evaluator = ModelEvaluator("Sistema de Análise de Sentimentos")
    
    # Esta função seria chamada após treinar o modelo
    # Por enquanto, simular resultados para demonstração
    
    print("Módulo de avaliação carregado com sucesso!")
    print("Para usar: instanciar ModelEvaluator e chamar evaluate_model()")
    
    return evaluator

if __name__ == "__main__":
    evaluate_sentiment_model()