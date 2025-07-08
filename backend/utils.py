import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def balance_dataset(df, class_column):
    """
    Balance dataset by upsampling minority classes
    
    Args:
        df (DataFrame): Input dataframe
        class_column (str): Column containing class labels
        
    Returns:
        DataFrame: Balanced dataframe
    """
    class_counts = df[class_column].value_counts()
    max_count = class_counts.max()
    
    print(f"Original class distribution:")
    print(class_counts)
    print(f"\nBalancing to {max_count} instances per class")
    
    # Oversample each class to match the majority class
    balanced_dfs = []
    for class_name in class_counts.index:
        class_df = df[df[class_column] == class_name]
        balanced_class_df = class_df.sample(n=max_count, replace=True, random_state=42)
        balanced_dfs.append(balanced_class_df)
    
    # Combine and shuffle
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nBalanced class distribution:")
    print(balanced_df[class_column].value_counts())
    
    return balanced_df

def create_visualizations(y_test, y_pred, classes, classification_report, confusion_mat):
    """
    Create and save visualization plots
    
    Args:
        y_test: True labels
        y_pred: Predicted labels  
        classes: Class names
        classification_report: Classification report dict
        confusion_mat: Confusion matrix
    """
    # Set up matplotlib for better looking plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    os.makedirs("visualizations", exist_ok=True)
    
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=classes,
                yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title('Matriz de Confusão - Classificação de Sentimentos', fontsize=16, fontweight='bold')
    plt.xlabel('Predição', fontsize=12)
    plt.ylabel('Valor Real', fontsize=12)
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Class Distribution
    plt.figure(figsize=(10, 6))
    unique, counts = np.unique(y_test, return_counts=True)
    colors = ['#8E44AD', '#F1C40F', '#E74C3C', '#2ECC71']  # Purple, Yellow, Red, Green
    bars = plt.bar(unique, counts, color=colors[:len(unique)])
    plt.title('Distribuição de Sentimentos no Dataset de Teste', fontsize=16, fontweight='bold')
    plt.xlabel('Sentimento', fontsize=12)
    plt.ylabel('Quantidade', fontsize=12)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. F1-Score and Accuracy per Class
    if classification_report and 'macro avg' in classification_report:
        classes_data = []
        for class_name in classes:
            if class_name in classification_report:
                classes_data.append({
                    'Classe': class_name,
                    'Precisão': classification_report[class_name]['precision'],
                    'Recall': classification_report[class_name]['recall'],
                    'F1-Score': classification_report[class_name]['f1-score']
                })
        
        if classes_data:
            df_metrics = pd.DataFrame(classes_data)
            
            # Create metrics visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            metrics = ['Precisão', 'Recall', 'F1-Score']
            colors = ['skyblue', 'lightcoral', 'lightgreen']
            
            for i, metric in enumerate(metrics):
                bars = axes[i].bar(df_metrics['Classe'], df_metrics[metric], color=colors[i])
                axes[i].set_title(f'{metric} por Classe', fontweight='bold')
                axes[i].set_ylabel(metric)
                axes[i].set_ylim(0, 1)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.3f}',
                                ha='center', va='bottom', fontweight='bold')
            
            plt.suptitle('Métricas de Performance por Classe', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('visualizations/class_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Create donut chart for analytics (matching prototype)
    create_donut_chart(y_test)
    
    print("Visualizations saved to 'visualizations/' directory")

def create_donut_chart(y_data):
    """Create donut chart matching the analytics prototype"""
    plt.figure(figsize=(10, 8))
    
    # Count sentiments
    unique, counts = np.unique(y_data, return_counts=True)
    total = sum(counts)
    percentages = [(count/total)*100 for count in counts]
    
    # Colors matching the prototype
    color_map = {
        'tristeza': '#8E44AD',  # Purple
        'alegria': '#F1C40F',   # Yellow  
        'raiva': '#E74C3C',     # Red
        'surpresa': '#2ECC71'   # Green
    }
    
    colors = [color_map.get(sentiment, '#95A5A6') for sentiment in unique]
    
    # Create donut chart
    wedges, texts, autotexts = plt.pie(percentages, 
                                      labels=[f'{s.title()}\n{p:.1f}%' for s, p in zip(unique, percentages)],
                                      colors=colors,
                                      autopct='',
                                      startangle=90,
                                      pctdistance=0.85)
    
    # Create donut hole
    centre_circle = plt.Circle((0,0), 0.60, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.title('Análise Gráfica: Distribuição de Sentimentos', fontsize=16, fontweight='bold', pad=20)
    
    # Create legend matching prototype
    legend_labels = []
    for sentiment, percentage in zip(unique, percentages):
        legend_labels.append(f'{sentiment.title()}: {percentage:.1f}%')
    
    plt.legend(wedges, legend_labels, title="Legenda", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('visualizations/donut_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def get_analytics_data(y_data):
    """Get analytics data for the frontend"""
    unique, counts = np.unique(y_data, return_counts=True)
    total = sum(counts)
    
    analytics = []
    color_map = {
        'tristeza': '#8E44AD',
        'alegria': '#F1C40F', 
        'raiva': '#E74C3C',
        'surpresa': '#2ECC71'
    }
    
    for sentiment, count in zip(unique, counts):
        percentage = (count / total) * 100
        analytics.append({
            'sentiment': sentiment.title(),
            'count': int(count),
            'percentage': round(percentage, 1),
            'color': color_map.get(sentiment, '#95A5A6')
        })
    
    return sorted(analytics, key=lambda x: x['percentage'], reverse=True)
