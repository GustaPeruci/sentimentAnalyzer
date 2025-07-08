"""
Configurações centralizadas do Zonalyze
Gerenciamento de configurações e variáveis de ambiente
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabaseConfig:
    """Configurações do banco de dados"""
    url: Optional[str] = None
    pool_size: int = 10
    max_retries: int = 3
    retry_delay: int = 2
    
    def __post_init__(self):
        self.url = os.environ.get('DATABASE_URL')

@dataclass
class MLConfig:
    """Configurações do modelo de Machine Learning"""
    model_path: str = "models/sentiment_model.pkl"
    bert_model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    max_length: int = 512
    batch_size: int = 32
    
    def __post_init__(self):
        
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
        except ImportError:
            pass

@dataclass
class AppConfig:
    """Configurações gerais da aplicação"""
    secret_key: str = "dev-secret-key"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 5000
    
    
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    
    
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        self.secret_key = os.environ.get('SESSION_SECRET', self.secret_key)
        self.debug = os.environ.get('FLASK_DEBUG', '').lower() in ['true', '1']


db_config = DatabaseConfig()
ml_config = MLConfig()
app_config = AppConfig()

def get_config():
    """Retorna todas as configurações"""
    return {
        'database': db_config,
        'ml': ml_config,
        'app': app_config
    }