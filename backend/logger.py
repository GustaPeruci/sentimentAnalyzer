"""
Sistema de logging centralizado para Zonalyze
Configuração avançada de logs com diferentes níveis e destinos
"""
import logging
import logging.handlers
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from .config import app_config

class ZonalyzeFormatter(logging.Formatter):
    """Formatter customizado para logs do Zonalyze"""
    
    def format(self, record):
        
        record.timestamp = datetime.utcnow().isoformat()
        record.app_name = "Zonalyze"
        
        
        formatted = super().format(record)
        
        
        if hasattr(record, 'context'):
            formatted += f" | Context: {json.dumps(record.context)}"
        
        return formatted

class StructuredLogger:
    """Logger estruturado para facilitar análise de logs"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.setup_logger()
    
    def setup_logger(self):
        """Configurar logger com handlers apropriados"""
        self.logger.setLevel(getattr(logging, app_config.log_level))
        
        
        if self.logger.handlers:
            return
        
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = ZonalyzeFormatter(
            '%(asctime)s - %(app_name)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        
        if not app_config.debug:
            try:
                os.makedirs('logs', exist_ok=True)
                file_handler = logging.handlers.RotatingFileHandler(
                    'logs/zonalyze.log',
                    maxBytes=10*1024*1024,  
                    backupCount=5
                )
                file_handler.setLevel(logging.DEBUG)
                file_formatter = ZonalyzeFormatter(app_config.log_format)
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.warning(f"Não foi possível criar arquivo de log: {e}")
    
    def log_api_request(self, endpoint: str, method: str, ip: str, 
                       status_code: int, response_time: float, 
                       additional_context: Optional[Dict[str, Any]] = None):
        """Log específico para requests da API"""
        context = {
            'type': 'api_request',
            'endpoint': endpoint,
            'method': method,
            'client_ip': ip,
            'status_code': status_code,
            'response_time_ms': round(response_time * 1000, 2),
            **(additional_context or {})
        }
        
        level = logging.ERROR if status_code >= 400 else logging.INFO
        self.logger.log(
            level,
            f"API {method} {endpoint} - {status_code} - {context['response_time_ms']}ms",
            extra={'context': context}
        )
    
    def log_sentiment_analysis(self, text_length: int, sentiment: str, 
                             confidence: float, processing_time: float,
                             model_used: str = "default"):
        """Log específico para análises de sentimento"""
        context = {
            'type': 'sentiment_analysis',
            'text_length': text_length,
            'predicted_sentiment': sentiment,
            'confidence_score': confidence,
            'processing_time_ms': round(processing_time * 1000, 2),
            'model_used': model_used
        }
        
        self.logger.info(
            f"Sentiment analysis: {sentiment} ({confidence:.3f}) - {context['processing_time_ms']}ms",
            extra={'context': context}
        )
    
    def log_database_operation(self, operation: str, table: str, 
                             success: bool, duration: float,
                             record_count: int = 1):
        """Log específico para operações de banco"""
        context = {
            'type': 'database_operation',
            'operation': operation,
            'table': table,
            'success': success,
            'duration_ms': round(duration * 1000, 2),
            'record_count': record_count
        }
        
        level = logging.INFO if success else logging.ERROR
        self.logger.log(
            level,
            f"DB {operation} on {table}: {'SUCCESS' if success else 'FAILED'} - {context['duration_ms']}ms",
            extra={'context': context}
        )
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log estruturado para erros"""
        error_context = {
            'type': 'error',
            'error_class': error.__class__.__name__,
            'error_message': str(error),
            **(context or {})
        }
        
        self.logger.error(
            f"Error: {error.__class__.__name__}: {str(error)}",
            extra={'context': error_context},
            exc_info=True
        )
    
    def info(self, message: str, **kwargs):
        """Log info com contexto opcional"""
        if kwargs:
            self.logger.info(message, extra={'context': kwargs})
        else:
            self.logger.info(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning com contexto opcional"""
        if kwargs:
            self.logger.warning(message, extra={'context': kwargs})
        else:
            self.logger.warning(message)
    
    def error(self, message: str, **kwargs):
        """Log error com contexto opcional"""
        if kwargs:
            self.logger.error(message, extra={'context': kwargs})
        else:
            self.logger.error(message)


api_logger = StructuredLogger('zonalyze.api')
ml_logger = StructuredLogger('zonalyze.ml')
db_logger = StructuredLogger('zonalyze.database')
app_logger = StructuredLogger('zonalyze.app')

def get_logger(name: str) -> StructuredLogger:
    """Factory function para criar loggers"""
    return StructuredLogger(f'zonalyze.{name}')