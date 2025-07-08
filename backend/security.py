"""
Módulo de segurança para Zonalyze
Implementa validações, sanitização e proteções de segurança
"""
import re
import hashlib
from typing import Optional, Dict, List
from flask import request
from .logger import get_logger

logger = get_logger('security')

class SecurityValidator:
    """Validador de segurança para entrada de dados"""
    
    # Padrões suspeitos de injeção
    INJECTION_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # XSS básico
        r'javascript\s*:',  # JavaScript protocol
        r'on\w+\s*=',  # Event handlers
        r'(union|select|insert|delete|update|drop)\s+',  # SQL injection
        r'(exec|eval|system|shell)\s*\(',  # Code injection
        r'\.\.\/|\.\.\\\\'  # Directory traversal
    ]
    
    # Headers suspeitos
    SUSPICIOUS_HEADERS = [
        'x-forwarded-for',
        'x-real-ip',
        'x-remote-addr'
    ]
    
    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """Sanitizar texto de entrada"""
        if not isinstance(text, str):
            return ""
        
        # Remove caracteres de controle
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Remove tags HTML suspeitas
        text = re.sub(r'<[^>]+>', '', text)
        
        # Limita comprimento
        if len(text) > 10000:
            text = text[:10000]
        
        return text.strip()
    
    @classmethod
    def check_injection_attempts(cls, text: str) -> List[str]:
        """Verificar tentativas de injeção"""
        threats = []
        text_lower = text.lower()
        
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                threats.append(f"Possible injection pattern: {pattern}")
        
        return threats
    
    @classmethod
    def validate_request_headers(cls, headers: Dict) -> List[str]:
        """Validar headers da requisição"""
        warnings = []
        
        # Verificar User-Agent suspeito
        user_agent = headers.get('User-Agent', '').lower()
        suspicious_agents = ['sqlmap', 'nikto', 'nmap', 'burp', 'curl', 'wget']
        
        for agent in suspicious_agents:
            if agent in user_agent:
                warnings.append(f"Suspicious User-Agent: {agent}")
        
        # Verificar headers de proxy
        for header in cls.SUSPICIOUS_HEADERS:
            if header in headers:
                warnings.append(f"Proxy header detected: {header}")
        
        return warnings
    
    @classmethod
    def get_client_fingerprint(cls) -> str:
        """Gerar fingerprint do cliente para tracking"""
        components = [
            request.environ.get('REMOTE_ADDR', ''),
            request.headers.get('User-Agent', ''),
            request.headers.get('Accept-Language', ''),
            request.headers.get('Accept-Encoding', '')
        ]
        
        fingerprint_data = '|'.join(components)
        return hashlib.md5(fingerprint_data.encode()).hexdigest()[:16]

class RateLimitTracker:
    """Tracker para rate limiting avançado"""
    
    def __init__(self):
        self.attempts = {}  # IP -> attempts data
        self.blocked_ips = set()
    
    def record_attempt(self, ip: str, endpoint: str, success: bool):
        """Registrar tentativa de acesso"""
        if ip not in self.attempts:
            self.attempts[ip] = {
                'total_requests': 0,
                'failed_requests': 0,
                'endpoints': {},
                'first_seen': None,
                'last_seen': None
            }
        
        data = self.attempts[ip]
        data['total_requests'] += 1
        data['last_seen'] = request.timestamp if hasattr(request, 'timestamp') else None
        
        if not success:
            data['failed_requests'] += 1
        
        if endpoint not in data['endpoints']:
            data['endpoints'][endpoint] = {'requests': 0, 'failures': 0}
        
        data['endpoints'][endpoint]['requests'] += 1
        if not success:
            data['endpoints'][endpoint]['failures'] += 1
        
        # Auto-bloqueio por comportamento suspeito
        if self._is_suspicious_behavior(data):
            self.blocked_ips.add(ip)
            logger.warning(f"IP {ip} blocked for suspicious behavior", 
                         ip=ip, data=data)
    
    def _is_suspicious_behavior(self, data: Dict) -> bool:
        """Detectar comportamento suspeito"""
        # Muitas requisições falhadas
        if data['failed_requests'] > 50:
            return True
        
        # Taxa de falha alta
        if data['total_requests'] > 20:
            failure_rate = data['failed_requests'] / data['total_requests']
            if failure_rate > 0.5:  # Mais de 50% de falhas
                return True
        
        return False
    
    def is_blocked(self, ip: str) -> bool:
        """Verificar se IP está bloqueado"""
        return ip in self.blocked_ips
    
    def get_stats(self, ip: str) -> Optional[Dict]:
        """Obter estatísticas de um IP"""
        return self.attempts.get(ip)

# Instâncias globais
security_validator = SecurityValidator()
rate_limit_tracker = RateLimitTracker()

def security_check(func):
    """Decorator para verificação de segurança"""
    def wrapper(*args, **kwargs):
        try:
            # Obter IP do cliente
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', 
                                          request.environ.get('REMOTE_ADDR', 'unknown'))
            
            # Verificar se IP está bloqueado
            if rate_limit_tracker.is_blocked(client_ip):
                logger.warning(f"Blocked IP attempted access: {client_ip}")
                return {'error': 'Access denied'}, 403
            
            # Validar headers
            header_warnings = security_validator.validate_request_headers(dict(request.headers))
            if header_warnings:
                logger.warning("Suspicious headers detected", 
                             ip=client_ip, warnings=header_warnings)
            
            # Executar função original
            result = func(*args, **kwargs)
            
            # Registrar tentativa bem-sucedida
            rate_limit_tracker.record_attempt(client_ip, request.endpoint, True)
            
            return result
            
        except Exception as e:
            # Registrar tentativa falhada
            if 'client_ip' in locals():
                rate_limit_tracker.record_attempt(client_ip, request.endpoint, False)
            
            logger.error("Security check failed", error=str(e))
            raise
    
    return wrapper