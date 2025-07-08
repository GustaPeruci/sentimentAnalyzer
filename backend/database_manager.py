"""
Gerenciador robusto de banco de dados PostgreSQL
Sistema com reconexão automática e pool de conexões
"""
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import time

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        """Inicializar gerenciador de banco com pool de conexões"""
        self.connection_pool = None
        self.database_url = os.environ.get('DATABASE_URL')
        self.max_retries = 3
        self.retry_delay = 2
        
        if not self.database_url:
            logger.warning("DATABASE_URL não encontrada, usando modo sem persistência")
            self.use_fallback = True
            # Armazenamento em memória como fallback
            self.memory_storage = []
            return
        else:
            self.use_fallback = False
        
        if not self.use_fallback:
            try:
                self.initialize_pool()
                self.create_tables()
            except Exception as e:
                logger.error(f"Erro ao conectar ao banco de dados: {e}")
                logger.warning("Fallback para modo sem persistência")
                self.use_fallback = True
                self.memory_storage = []
    
    def initialize_pool(self):
        """Criar pool de conexões PostgreSQL"""
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 10,  # min e max conexões
                self.database_url,
                cursor_factory=RealDictCursor
            )
            logger.info("Pool de conexões PostgreSQL inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao criar pool de conexões: {e}")
            raise
    
    def get_connection(self):
        """Obter conexão do pool com retry automático"""
        for attempt in range(self.max_retries):
            try:
                if self.connection_pool:
                    conn = self.connection_pool.getconn()
                    if conn:
                        # Testar conexão
                        with conn.cursor() as cursor:
                            cursor.execute("SELECT 1")
                        return conn
                    else:
                        raise Exception("Não foi possível obter conexão do pool")
                else:
                    raise Exception("Pool de conexões não inicializado")
                    
            except Exception as e:
                logger.warning(f"Tentativa {attempt + 1} falhou: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    # Tentar reinicializar pool
                    try:
                        self.initialize_pool()
                    except:
                        pass
                else:
                    logger.error("Todas as tentativas de conexão falharam")
                    raise
    
    def return_connection(self, conn):
        """Retornar conexão para o pool"""
        if self.connection_pool and conn:
            self.connection_pool.putconn(conn)
    
    def create_tables(self):
        """Criar tabelas necessárias"""
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # Tabela para análises de sentimento
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sentiment_analyses (
                        id SERIAL PRIMARY KEY,
                        text TEXT NOT NULL,
                        sentiment VARCHAR(50) NOT NULL,
                        sentiment_key VARCHAR(50) NOT NULL,
                        confidence FLOAT NOT NULL,
                        probabilities JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ip_address VARCHAR(45),
                        user_agent VARCHAR(500)
                    )
                """)
                
                # Tabela para cache de analytics
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analytics_cache (
                        id SERIAL PRIMARY KEY,
                        cache_key VARCHAR(100) UNIQUE NOT NULL,
                        data JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP NOT NULL
                    )
                """)
                
                # Índices para performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sentiment_created_at 
                    ON sentiment_analyses(created_at DESC)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sentiment_key 
                    ON sentiment_analyses(sentiment_key)
                """)
                
                conn.commit()
                logger.info("Tabelas criadas/verificadas com sucesso")
                
        except Exception as e:
            logger.error(f"Erro ao criar tabelas: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.return_connection(conn)
    
    def save_sentiment_analysis(self, text: str, sentiment: str, sentiment_key: str, 
                               confidence: float, probabilities: Dict, 
                               ip_address: Optional[str] = None, 
                               user_agent: Optional[str] = None) -> bool:
        """Salvar análise de sentimento no banco"""
        if self.use_fallback:
            return self._save_sentiment_fallback(text, sentiment, sentiment_key, confidence, probabilities, ip_address, user_agent)
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO sentiment_analyses 
                    (text, sentiment, sentiment_key, confidence, probabilities, created_at, ip_address, user_agent)
                    VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s)
                """, (text, sentiment, sentiment_key, confidence, 
                     json.dumps(probabilities), ip_address, user_agent))
                conn.commit()
                logger.info(f"Análise salva: {sentiment} ({confidence:.3f})")
                return True
                
        except Exception as e:
            logger.error(f"Erro ao salvar análise: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self.return_connection(conn)
    
    def get_sentiment_history(self, limit: int = 50) -> List[Dict]:
        """Obter histórico de análises"""
        if self.use_fallback:
            return self.memory_storage[-limit:] if len(self.memory_storage) > limit else self.memory_storage
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT text, sentiment, sentiment_key, confidence, 
                           probabilities, created_at, ip_address
                    FROM sentiment_analyses 
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (limit,))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'text': row['text'],
                        'sentiment': row['sentiment'],
                        'sentiment_key': row['sentiment_key'],
                        'confidence': row['confidence'],
                        'probabilities': row['probabilities'],
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                        'ip_address': row['ip_address']
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Erro ao buscar histórico: {e}")
            return []
        finally:
            if conn:
                self.return_connection(conn)
    
    def get_sentiment_analytics(self, limit: int = 1000) -> Dict:
        """Obter dados de analytics"""
        if self.use_fallback:
            return self._get_analytics_fallback(limit)
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # Contar por tipo de sentimento
                cursor.execute("""
                    SELECT sentiment_key, sentiment, COUNT(*) as count
                    FROM sentiment_analyses 
                    WHERE created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY sentiment_key, sentiment
                    ORDER BY count DESC
                """, (30,))  # Últimos 30 dias
                
                sentiment_counts = {}
                total = 0
                for row in cursor.fetchall():
                    sentiment_counts[row['sentiment_key']] = {
                        'count': row['count'],
                        'label': row['sentiment']
                    }
                    total += row['count']
                
                # Calcular percentuais
                analytics = []
                colors = {
                    'alegria': '#28a745',
                    'tristeza': '#6c757d', 
                    'raiva': '#dc3545',
                    'surpresa': '#ffc107'
                }
                
                for key, data in sentiment_counts.items():
                    analytics.append({
                        'sentiment': data['label'],
                        'count': data['count'],
                        'percentage': round((data['count'] / total * 100), 1) if total > 0 else 0,
                        'color': colors.get(key, '#007bff')
                    })
                
                return {
                    'analytics': analytics,
                    'total_analyses': total
                }
                
        except Exception as e:
            logger.error(f"Erro ao buscar analytics: {e}")
            return {'analytics': [], 'total_analyses': 0}
        finally:
            if conn:
                self.return_connection(conn)
    
    def get_sentiment_count(self) -> int:
        """Obter total de análises"""
        if self.use_fallback:
            return len(self.memory_storage)
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM sentiment_analyses")
                result = cursor.fetchone()
                return result[0] if result else 0
                
        except Exception as e:
            logger.error(f"Erro ao contar análises: {e}")
            return 0
        finally:
            if conn:
                self.return_connection(conn)
    
    def clear_sentiment_history(self) -> bool:
        """Limpar histórico de análises"""
        if self.use_fallback:
            self.memory_storage.clear()
            return True
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM sentiment_analyses")
                cursor.execute("DELETE FROM analytics_cache")
                conn.commit()
                logger.info("Histórico limpo com sucesso")
                return True
                
        except Exception as e:
            logger.error(f"Erro ao limpar histórico: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self.return_connection(conn)
    
    def health_check(self) -> Dict[str, Any]:
        """Verificar saúde do banco de dados"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT version()")
                version = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM sentiment_analyses")
                total_records = cursor.fetchone()[0]
                
            self.return_connection(conn)
            
            return {
                'status': 'healthy',
                'version': version,
                'total_records': total_records,
                'pool_size': self.connection_pool.closed if self.connection_pool else 0
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def _save_sentiment_fallback(self, text: str, sentiment: str, sentiment_key: str, 
                                confidence: float, probabilities: Dict, 
                                ip_address: Optional[str] = None, 
                                user_agent: Optional[str] = None) -> bool:
        """Salvar análise em memória como fallback"""
        try:
            analysis = {
                'id': len(self.memory_storage) + 1,
                'text': text,
                'sentiment': sentiment,
                'sentiment_key': sentiment_key,
                'confidence': confidence,
                'probabilities': probabilities,
                'created_at': datetime.now().isoformat(),
                'ip_address': ip_address,
                'user_agent': user_agent
            }
            self.memory_storage.append(analysis)
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar análise em memória: {e}")
            return False

    def _get_analytics_fallback(self, limit: int = 1000) -> Dict:
        """Obter analytics de dados em memória"""
        try:
            data = self.memory_storage[-limit:] if len(self.memory_storage) > limit else self.memory_storage
            
            if not data:
                return {
                    'total_analyses': 0,
                    'sentiment_distribution': {},
                    'avg_confidence': 0,
                    'recent_activity': []
                }
            
            # Calcular distribuição de sentimentos
            sentiment_dist = {}
            total_confidence = 0
            
            for analysis in data:
                sentiment = analysis['sentiment_key']
                sentiment_dist[sentiment] = sentiment_dist.get(sentiment, 0) + 1
                total_confidence += analysis['confidence']
            
            return {
                'total_analyses': len(data),
                'sentiment_distribution': sentiment_dist,
                'avg_confidence': total_confidence / len(data) if data else 0,
                'recent_activity': data[-10:]  # Últimas 10 análises
            }
        except Exception as e:
            logger.error(f"Erro ao obter analytics em memória: {e}")
            return {
                'total_analyses': 0,
                'sentiment_distribution': {},
                'avg_confidence': 0,
                'recent_activity': []
            }

    def close(self):
        """Fechar pool de conexões"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Pool de conexões fechado")

# Instância global
db_manager = None

def get_db_manager():
    """Obter instância do gerenciador de banco"""
    global db_manager
    if db_manager is None:
        try:
            db_manager = DatabaseManager()
        except Exception as e:
            logger.error(f"Falha ao inicializar DatabaseManager: {e}")
            db_manager = None
    return db_manager