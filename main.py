"""
Zonalyze - Sistema de Análise de Sentimentos
Ponto de entrada principal da aplicação
"""
from flask import Flask, render_template, send_from_directory, request
from flask_cors import CORS
from backend.api import api_bp
from backend.database_manager import get_db_manager
from backend.config import app_config
from backend.logger import app_logger
from backend.performance_monitor import performance_monitor
import os
import time

# Criar aplicação Flask
app = Flask(__name__, 
           static_folder='src', 
           static_url_path='')

# Configurar CORS
CORS(app, origins=["*"])

# Configurações usando config centralizado
app.config['SECRET_KEY'] = app_config.secret_key
app.config['DATABASE_URL'] = os.environ.get('DATABASE_URL')
app.config['DEBUG'] = app_config.debug

# Middleware para logging de requests
@app.before_request
def log_request_info():
    start_time = time.time()
    request.start_time = start_time

@app.after_request
def log_response_info(response):
    if hasattr(request, 'start_time'):
        response_time = time.time() - request.start_time
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', 
                                      request.environ.get('REMOTE_ADDR', 'unknown'))
        
        # Log request usando performance monitor
        performance_monitor.record_api_request(
            request.endpoint or 'unknown',
            request.method,
            response_time,
            response.status_code
        )
        
        # Log detalhado para analytics
        app_logger.log_api_request(
            request.endpoint or request.path,
            request.method,
            client_ip,
            response.status_code,
            response_time
        )
    
    return response

# Registrar blueprints
app.register_blueprint(api_bp, url_prefix='/api')

# Inicializar banco de dados
db_manager = get_db_manager()

@app.route('/')
def index():
    """Servir a aplicação web principal"""
    return send_from_directory('.', 'simple_app.html')

@app.route('/health')
def health():
    """Endpoint de verificação de saúde avançado"""
    health_status = performance_monitor.get_health_status()
    
    # Adicionar informações específicas do Zonalyze
    health_status.update({
        'service': 'Zonalyze',
        'version': '2.0.0',
        'timestamp': time.time()
    })
    
    # Verificar banco de dados
    if db_manager:
        db_health = db_manager.health_check()
        health_status['database'] = db_health
    else:
        health_status['database'] = {'status': 'unavailable', 'mode': 'fallback'}
    
    return health_status

@app.route('/metrics')
def metrics():
    """Endpoint de métricas para monitoramento"""
    return {
        'api_metrics': performance_monitor.get_api_summary(),
        'system_metrics': performance_monitor.get_system_summary(),
        'recent_activity': performance_monitor.get_recent_activity(30)
    }

@app.route('/<path:path>')
def static_files(path):
    """Servir arquivos estáticos do Angular"""
    return send_from_directory('src', path)

@app.errorhandler(404)
def not_found(e):
    """Redirecionar 404s para a aplicação Angular"""
    return send_from_directory('src', 'index.html')

if __name__ == '__main__':
    app_logger.info("Iniciando Zonalyze - Sistema de Análise de Sentimentos v2.0")
    app_logger.info(f"Configurações: Debug={app_config.debug}, Host={app_config.host}, Port={app_config.port}")
    app.run(host=app_config.host, port=app_config.port, debug=app_config.debug)
