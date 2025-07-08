"""
Rate Limiter para proteger APIs do Zonalyze
Previne abuso e garante estabilidade do serviço
"""
from functools import wraps
from collections import defaultdict, deque
from datetime import datetime, timedelta
from flask import request, jsonify
import threading

class RateLimiter:
    """Rate limiter baseado em IP e janela deslizante"""
    
    def __init__(self):
        self.requests = defaultdict(deque)  # IP -> lista de timestamps
        self.lock = threading.Lock()
    
    def is_allowed(self, ip: str, max_requests: int, window_minutes: int) -> tuple[bool, dict]:
        """
        Verifica se o IP pode fazer uma request
        
        Returns:
            tuple: (allowed: bool, info: dict)
        """
        now = datetime.now()
        window_start = now - timedelta(minutes=window_minutes)
        
        with self.lock:
            # Remove requests antigas
            while self.requests[ip] and self.requests[ip][0] < window_start:
                self.requests[ip].popleft()
            
            current_count = len(self.requests[ip])
            
            if current_count >= max_requests:
                # Rate limit excedido
                oldest_request = self.requests[ip][0] if self.requests[ip] else now
                reset_time = oldest_request + timedelta(minutes=window_minutes)
                
                return False, {
                    'current_count': current_count,
                    'max_requests': max_requests,
                    'window_minutes': window_minutes,
                    'reset_time': reset_time.isoformat(),
                    'retry_after_seconds': int((reset_time - now).total_seconds())
                }
            
            # Adicionar nova request
            self.requests[ip].append(now)
            
            return True, {
                'current_count': current_count + 1,
                'max_requests': max_requests,
                'window_minutes': window_minutes,
                'remaining': max_requests - (current_count + 1)
            }

# Instância global
rate_limiter = RateLimiter()

def rate_limit(max_requests: int = 60, window_minutes: int = 1):
    """
    Decorator para aplicar rate limiting
    
    Args:
        max_requests: Número máximo de requests
        window_minutes: Janela de tempo em minutos
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Obter IP do cliente
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', 
                                          request.environ.get('REMOTE_ADDR', 'unknown'))
            
            if client_ip != 'unknown' and ',' in client_ip:
                client_ip = client_ip.split(',')[0].strip()
            
            allowed, info = rate_limiter.is_allowed(client_ip, max_requests, window_minutes)
            
            if not allowed:
                response = jsonify({
                    'error': 'Rate limit exceeded',
                    'message': f'Too many requests. Try again in {info["retry_after_seconds"]} seconds.',
                    'rate_limit': info
                })
                response.status_code = 429
                response.headers['Retry-After'] = str(info['retry_after_seconds'])
                response.headers['X-RateLimit-Limit'] = str(max_requests)
                response.headers['X-RateLimit-Remaining'] = '0'
                response.headers['X-RateLimit-Reset'] = info['reset_time']
                return response
            
            # Adicionar headers informativos
            response = f(*args, **kwargs)
            if hasattr(response, 'headers'):
                response.headers['X-RateLimit-Limit'] = str(max_requests)
                response.headers['X-RateLimit-Remaining'] = str(info['remaining'])
                
            return response
        return wrapper
    return decorator