# Zonalyze - Análise de Sentimentos

Zonalyze é uma aplicação web completa para análise de sentimentos de comentários usando inteligência artificial. O sistema utiliza embeddings BERT e regressão logística para classificar sentimentos em português.

## 🚀 Características

- **Backend Python**: Flask API com embeddings BERT (all-MiniLM-L6-v2)
- **Frontend Angular**: Interface moderna e responsiva em português
- **Análise de Sentimentos**: Classificação em 4 categorias (Tristeza, Alegria, Raiva, Surpresa)
- **Visualizações**: Gráficos de pizza e barras com Chart.js
- **Cache Inteligente**: Sistema de cache para embeddings BERT (.npy)
- **Offline**: Funciona completamente offline após treinamento

## 📋 Pré-requisitos

### Backend
- Python 3.12+
- pip

### Frontend
- Node.js 16+
- npm ou yarn
- Angular CLI

## 🛠️ Instalação

### 1. Configuração do Backend

```bash
# Instalar dependências Python
pip install flask flask-cors pandas numpy scikit-learn matplotlib seaborn tqdm sentence-transformers torch

# Navegar para o diretório do projeto
cd zonalyze

# Criar diretórios necessários
mkdir -p models visualizations
