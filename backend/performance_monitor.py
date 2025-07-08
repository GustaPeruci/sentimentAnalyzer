"""
Monitor de performance para Zonalyze
Coleta métricas de desempenho do sistema e APIs
"""
import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .logger import get_logger

logger = get_logger('performance')

class PerformanceMonitor:
    """Monitor de performance com métricas em tempo real"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.api_metrics = defaultdict(lambda: {
            'response_times': deque(maxlen=1000),
            'request_count': 0,
            'error_count': 0,
            'last_request': None
        })
        self.system_metrics = {
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'disk_usage': deque(maxlen=100)
        }
        self.start_time = datetime.now()
        self.lock = threading.Lock()
        
        # Iniciar coleta de métricas do sistema
        self._start_system_monitoring()
    
    def _start_system_monitoring(self):
        """Iniciar thread para monitoramento do sistema"""
        def collect_system_metrics():
            while True:
                try:
                    # Tentar importar psutil, mas funcionar sem ele se necessário
                    try:
                        import psutil
                        cpu_percent = psutil.cpu_percent(interval=1)
                        memory = psutil.virtual_memory()
                        disk = psutil.disk_usage('/')
                        
                        with self.lock:
                            self.system_metrics['cpu_usage'].append({
                                'timestamp': datetime.now(),
                                'value': cpu_percent
                            })
                            self.system_metrics['memory_usage'].append({
                                'timestamp': datetime.now(),
                                'value': memory.percent
                            })
                            self.system_metrics['disk_usage'].append({
                                'timestamp': datetime.now(),
                                'value': disk.percent
                            })
                    except ImportError:
                        # Fallback para métricas básicas se psutil não estiver disponível
                        with self.lock:
                            self.system_metrics['cpu_usage'].append({
                                'timestamp': datetime.now(),
                                'value': 0  # Placeholder
                            })
                            self.system_metrics['memory_usage'].append({
                                'timestamp': datetime.now(),
                                'value': 0  # Placeholder
                            })
                            self.system_metrics['disk_usage'].append({
                                'timestamp': datetime.now(),
                                'value': 0  # Placeholder
                            })
                except Exception as e:
                    logger.error(f"Erro ao coletar métricas do sistema: {e}")
                
                time.sleep(60)  # Coletar a cada minuto
        
        monitor_thread = threading.Thread(target=collect_system_metrics, daemon=True)
        monitor_thread.start()
    
    def record_api_request(self, endpoint: str, method: str, response_time: float, 
                          status_code: int, error: Optional[str] = None):
        """Registrar métrica de request da API"""
        with self.lock:
            endpoint_key = f"{method}:{endpoint}"
            metrics = self.api_metrics[endpoint_key]
            
            metrics['response_times'].append({
                'timestamp': datetime.now(),
                'response_time': response_time,
                'status_code': status_code
            })
            metrics['request_count'] += 1
            
            if status_code >= 400:
                metrics['error_count'] += 1
            
            metrics['last_request'] = datetime.now()
            
            # Log métricas críticas
            if response_time > 5.0:  # Mais de 5 segundos
                logger.warning(f"Slow API response: {endpoint_key} took {response_time:.2f}s")
            
            if error:
                logger.error(f"API error on {endpoint_key}: {error}")
    
    def record_ml_prediction(self, model_name: str, prediction_time: float, 
                           text_length: int, confidence: float):
        """Registrar métrica de predição ML"""
        with self.lock:
            ml_metrics = self.api_metrics[f"ml:{model_name}"]
            ml_metrics['response_times'].append({
                'timestamp': datetime.now(),
                'prediction_time': prediction_time,
                'text_length': text_length,
                'confidence': confidence
            })
            ml_metrics['request_count'] += 1
            ml_metrics['last_request'] = datetime.now()
    
    def get_api_summary(self) -> Dict:
        """Obter resumo das métricas da API"""
        with self.lock:
            summary = {}
            
            for endpoint, metrics in self.api_metrics.items():
                if metrics['request_count'] == 0:
                    continue
                
                response_times = [r['response_time'] for r in metrics['response_times'] 
                                if 'response_time' in r]
                
                if response_times:
                    avg_response_time = sum(response_times) / len(response_times)
                    max_response_time = max(response_times)
                    min_response_time = min(response_times)
                else:
                    avg_response_time = max_response_time = min_response_time = 0
                
                error_rate = (metrics['error_count'] / metrics['request_count']) * 100
                
                summary[endpoint] = {
                    'request_count': metrics['request_count'],
                    'error_count': metrics['error_count'],
                    'error_rate': round(error_rate, 2),
                    'avg_response_time': round(avg_response_time, 3),
                    'max_response_time': round(max_response_time, 3),
                    'min_response_time': round(min_response_time, 3),
                    'last_request': metrics['last_request'].isoformat() if metrics['last_request'] else None
                }
            
            return summary
    
    def get_system_summary(self) -> Dict:
        """Obter resumo das métricas do sistema"""
        with self.lock:
            uptime = datetime.now() - self.start_time
            
            def get_metric_stats(metric_name):
                values = [m['value'] for m in self.system_metrics[metric_name]]
                if not values:
                    return {'current': 0, 'avg': 0, 'max': 0}
                
                return {
                    'current': values[-1] if values else 0,
                    'avg': round(sum(values) / len(values), 2),
                    'max': round(max(values), 2)
                }
            
            return {
                'uptime_seconds': int(uptime.total_seconds()),
                'uptime_human': str(uptime).split('.')[0],  # Remove microseconds
                'cpu': get_metric_stats('cpu_usage'),
                'memory': get_metric_stats('memory_usage'),
                'disk': get_metric_stats('disk_usage'),
                'total_requests': sum(m['request_count'] for m in self.api_metrics.values()),
                'total_errors': sum(m['error_count'] for m in self.api_metrics.values())
            }
    
    def get_recent_activity(self, minutes: int = 30) -> Dict:
        """Obter atividade recente"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.lock:
            recent_requests = 0
            recent_errors = 0
            
            for metrics in self.api_metrics.values():
                for request in metrics['response_times']:
                    if request['timestamp'] > cutoff_time:
                        recent_requests += 1
                        if request.get('status_code', 200) >= 400:
                            recent_errors += 1
            
            return {
                'time_window_minutes': minutes,
                'recent_requests': recent_requests,
                'recent_errors': recent_errors,
                'requests_per_minute': round(recent_requests / minutes, 2),
                'error_rate': round((recent_errors / recent_requests * 100) if recent_requests > 0 else 0, 2)
            }
    
    def get_health_status(self) -> Dict:
        """Verificar status de saúde do sistema"""
        system_summary = self.get_system_summary()
        recent_activity = self.get_recent_activity()
        
        # Critérios de saúde
        health_issues = []
        
        if system_summary['cpu']['current'] > 80:
            health_issues.append("High CPU usage")
        
        if system_summary['memory']['current'] > 85:
            health_issues.append("High memory usage")
        
        if system_summary['disk']['current'] > 90:
            health_issues.append("High disk usage")
        
        if recent_activity['error_rate'] > 10:
            health_issues.append("High error rate")
        
        status = "healthy" if not health_issues else "warning" if len(health_issues) < 3 else "critical"
        
        return {
            'status': status,
            'issues': health_issues,
            'last_check': datetime.now().isoformat(),
            'system': system_summary,
            'recent_activity': recent_activity
        }

# Instância global
performance_monitor = PerformanceMonitor()

def monitor_performance(endpoint: str, method: str = "GET"):
    """Decorator para monitorar performance de endpoints"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            error = None
            status_code = 200
            
            try:
                result = func(*args, **kwargs)
                
                # Tentar extrair status code da resposta
                if hasattr(result, 'status_code'):
                    status_code = result.status_code
                elif isinstance(result, tuple) and len(result) >= 2:
                    status_code = result[1]
                
                return result
            
            except Exception as e:
                error = str(e)
                status_code = 500
                raise
            
            finally:
                response_time = time.time() - start_time
                performance_monitor.record_api_request(
                    endpoint, method, response_time, status_code, error
                )
        
        return wrapper
    return decorator