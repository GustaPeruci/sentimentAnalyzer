"""
Sistema de monitoramento e métricas para Zonalyze
Coleta e analisa métricas de performance em tempo real
"""
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import threading
import json

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Coletor de métricas em tempo real"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.request_times = deque(maxlen=max_history)
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.sentiment_stats = defaultdict(int)
        self.confidence_scores = deque(maxlen=max_history)
        self.start_time = datetime.utcnow()
        self._lock = threading.Lock()
    
    def record_request(self, endpoint: str, response_time: float, status_code: int):
        """Registrar métricas de request"""
        with self._lock:
            self.request_times.append({
                'endpoint': endpoint,
                'response_time': response_time,
                'status_code': status_code,
                'timestamp': datetime.utcnow()
            })
            self.request_counts[endpoint] += 1
            
            if status_code >= 400:
                self.error_counts[endpoint] += 1
    
    def record_sentiment_analysis(self, sentiment: str, confidence: float):
        """Registrar análise de sentimento"""
        with self._lock:
            self.sentiment_stats[sentiment] += 1
            self.confidence_scores.append(confidence)
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Obter métricas em tempo real"""
        with self._lock:
            current_time = datetime.utcnow()
            uptime = current_time - self.start_time
            
            
            recent_requests = [
                req for req in self.request_times 
                if current_time - req['timestamp'] <= timedelta(minutes=5)
            ]
            
            response_times = [req['response_time'] for req in recent_requests]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            
            total_requests = sum(self.request_counts.values())
            total_errors = sum(self.error_counts.values())
            error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
            
            
            avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0
            
            return {
                'system': {
                    'uptime_seconds': uptime.total_seconds(),
                    'uptime_human': str(uptime).split('.')[0],
                    'total_requests': total_requests,
                    'total_errors': total_errors,
                    'error_rate_percent': round(error_rate, 2)
                },
                'performance': {
                    'avg_response_time_ms': round(avg_response_time * 1000, 2),
                    'requests_last_5min': len(recent_requests),
                    'avg_confidence_score': round(avg_confidence, 3)
                },
                'sentiment_distribution': dict(self.sentiment_stats),
                'endpoint_usage': dict(self.request_counts),
                'timestamp': current_time.isoformat()
            }
    
    def get_health_score(self) -> Dict[str, Any]:
        """Calcular score de saúde do sistema"""
        metrics = self.get_real_time_metrics()
        
        
        health_score = 100
        issues = []
        
        
        if metrics['system']['error_rate_percent'] > 5:
            health_score -= 30
            issues.append(f"Alta taxa de erro: {metrics['system']['error_rate_percent']}%")
        
        
        if metrics['performance']['avg_response_time_ms'] > 1000:
            health_score -= 20
            issues.append(f"Tempo de resposta alto: {metrics['performance']['avg_response_time_ms']}ms")
        
        
        if metrics['performance']['avg_confidence_score'] < 0.6:
            health_score -= 25
            issues.append(f"Confiança baixa: {metrics['performance']['avg_confidence_score']}")
        
        
        if health_score >= 90:
            status = 'excellent'
        elif health_score >= 70:
            status = 'good'
        elif health_score >= 50:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'health_score': max(0, health_score),
            'status': status,
            'issues': issues,
            'recommendations': self._get_recommendations(issues)
        }
    
    def _get_recommendations(self, issues: List[str]) -> List[str]:
        """Gerar recomendações baseadas nos problemas"""
        recommendations = []
        
        for issue in issues:
            if 'taxa de erro' in issue:
                recommendations.append("Verificar logs de erro e corrigir problemas de validação")
            elif 'tempo de resposta' in issue:
                recommendations.append("Otimizar consultas ao banco de dados e cache")
            elif 'confiança baixa' in issue:
                recommendations.append("Revisar modelo de análise de sentimentos e dados de treinamento")
        
        return recommendations


metrics_collector = MetricsCollector()

def track_request(endpoint: str):
    """Decorator para rastrear requests"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                response_time = time.time() - start_time
                metrics_collector.record_request(endpoint, response_time, 200)
                return result
            except Exception as e:
                response_time = time.time() - start_time
                metrics_collector.record_request(endpoint, response_time, 500)
                raise
        return wrapper
    return decorator

def track_sentiment(sentiment: str, confidence: float):
    """Registrar análise de sentimento"""
    metrics_collector.record_sentiment_analysis(sentiment, confidence)

def get_monitoring_dashboard() -> Dict[str, Any]:
    """Obter dados completos do dashboard de monitoramento"""
    metrics = metrics_collector.get_real_time_metrics()
    health = metrics_collector.get_health_score()
    
    return {
        'metrics': metrics,
        'health': health,
        'alerts': _check_alerts(metrics, health),
        'dashboard_updated': datetime.utcnow().isoformat()
    }

def _check_alerts(metrics: Dict, health: Dict) -> List[Dict]:
    """Verificar se há alertas críticos"""
    alerts = []
    
    
    if metrics['system']['error_rate_percent'] > 10:
        alerts.append({
            'level': 'critical',
            'message': f"Taxa de erro crítica: {metrics['system']['error_rate_percent']}%",
            'action': 'Verificar logs imediatamente'
        })
    
    
    if metrics['performance']['avg_response_time_ms'] > 2000:
        alerts.append({
            'level': 'warning',
            'message': f"Performance degradada: {metrics['performance']['avg_response_time_ms']}ms",
            'action': 'Investigar causa da lentidão'
        })
    
    
    if metrics['performance']['avg_confidence_score'] < 0.5:
        alerts.append({
            'level': 'warning',
            'message': f"Confiança do modelo muito baixa: {metrics['performance']['avg_confidence_score']}",
            'action': 'Revisar modelo de ML'
        })
    
    return alerts