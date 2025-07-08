"""
Sistema de cache para Zonalyze
Cache inteligente para embeddings e predições
"""
import pickle
import hashlib
import os
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
from .logger import get_logger

logger = get_logger('cache')

class CacheManager:
    """Gerenciador de cache para embeddings e predições"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.embeddings_cache = {}  
        self.predictions_cache = {}  
        self.max_memory_items = 1000  
        
        
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "embeddings"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "predictions"), exist_ok=True)
        
        logger.info(f"Cache manager initialized: {cache_dir}")
    
    def _generate_key(self, text: str) -> str:
        """Gerar chave única para o texto"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_file_path(self, cache_type: str, key: str) -> str:
        """Obter caminho do arquivo de cache"""
        return os.path.join(self.cache_dir, cache_type, f"{key}.pkl")
    
    def save_embedding(self, text: str, embedding: np.ndarray) -> bool:
        """Salvar embedding no cache"""
        try:
            key = self._generate_key(text)
            
            
            if len(self.embeddings_cache) < self.max_memory_items:
                self.embeddings_cache[key] = {
                    'embedding': embedding,
                    'timestamp': time.time(),
                    'text_length': len(text)
                }
            
            
            file_path = self._get_file_path("embeddings", key)
            cache_data = {
                'embedding': embedding,
                'text': text[:100],  
                'timestamp': time.time(),
                'text_length': len(text)
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.debug(f"Embedding cached: {key[:8]}... (length: {len(text)})")
            return True
            
        except Exception as e:
            logger.error(f"Error saving embedding to cache: {e}")
            return False
    
    def load_embedding(self, text: str) -> Optional[np.ndarray]:
        """Carregar embedding do cache"""
        try:
            key = self._generate_key(text)
            
            
            if key in self.embeddings_cache:
                cache_item = self.embeddings_cache[key]
                logger.debug(f"Embedding found in memory cache: {key[:8]}...")
                return cache_item['embedding']
            
            
            file_path = self._get_file_path("embeddings", key)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                embedding = cache_data['embedding']
                
                
                if len(self.embeddings_cache) < self.max_memory_items:
                    self.embeddings_cache[key] = cache_data
                
                logger.debug(f"Embedding loaded from disk cache: {key[:8]}...")
                return embedding
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading embedding from cache: {e}")
            return None
    
    def save_prediction(self, text: str, prediction: Dict, model_version: str = "v1.0") -> bool:
        """Salvar predição no cache"""
        try:
            
            text_key = self._generate_key(text)
            key = f"{text_key}_{model_version}"
            
            
            if len(self.predictions_cache) < self.max_memory_items:
                self.predictions_cache[key] = {
                    'prediction': prediction,
                    'timestamp': time.time(),
                    'model_version': model_version,
                    'text_length': len(text)
                }
            
            
            file_path = self._get_file_path("predictions", key)
            cache_data = {
                'prediction': prediction,
                'text': text[:100],  
                'timestamp': time.time(),
                'model_version': model_version,
                'text_length': len(text)
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.debug(f"Prediction cached: {key[:8]}... (model: {model_version})")
            return True
            
        except Exception as e:
            logger.error(f"Error saving prediction to cache: {e}")
            return False
    
    def load_prediction(self, text: str, model_version: str = "v1.0") -> Optional[Dict]:
        """Carregar predição do cache"""
        try:
            text_key = self._generate_key(text)
            key = f"{text_key}_{model_version}"
            
            
            if key in self.predictions_cache:
                cache_item = self.predictions_cache[key]
                
                if time.time() - cache_item['timestamp'] < 86400:
                    logger.debug(f"Prediction found in memory cache: {key[:8]}...")
                    return cache_item['prediction']
                else:
                    
                    del self.predictions_cache[key]
            
            
            file_path = self._get_file_path("predictions", key)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                
                if time.time() - cache_data['timestamp'] < 86400:
                    prediction = cache_data['prediction']
                    
                    
                    if len(self.predictions_cache) < self.max_memory_items:
                        self.predictions_cache[key] = cache_data
                    
                    logger.debug(f"Prediction loaded from disk cache: {key[:8]}...")
                    return prediction
                else:
                    
                    os.remove(file_path)
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading prediction from cache: {e}")
            return None
    
    def get_cache_stats(self) -> Dict:
        """Obter estatísticas do cache"""
        memory_embeddings = len(self.embeddings_cache)
        memory_predictions = len(self.predictions_cache)
        
        
        disk_embeddings = len([f for f in os.listdir(os.path.join(self.cache_dir, "embeddings")) 
                              if f.endswith('.pkl')])
        disk_predictions = len([f for f in os.listdir(os.path.join(self.cache_dir, "predictions")) 
                               if f.endswith('.pkl')])
        
        
        cache_size = 0
        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    cache_size += os.path.getsize(file_path)
        
        return {
            'memory': {
                'embeddings': memory_embeddings,
                'predictions': memory_predictions,
                'total': memory_embeddings + memory_predictions,
                'max_items': self.max_memory_items
            },
            'disk': {
                'embeddings': disk_embeddings,
                'predictions': disk_predictions,
                'total': disk_embeddings + disk_predictions,
                'size_bytes': cache_size,
                'size_mb': round(cache_size / (1024 * 1024), 2)
            }
        }
    
    def clear_cache(self, cache_type: Optional[str] = None) -> bool:
        """Limpar cache"""
        try:
            if cache_type == "memory" or cache_type is None:
                self.embeddings_cache.clear()
                self.predictions_cache.clear()
                logger.info("Memory cache cleared")
            
            if cache_type == "disk" or cache_type is None:
                for cache_subdir in ["embeddings", "predictions"]:
                    cache_path = os.path.join(self.cache_dir, cache_subdir)
                    for file in os.listdir(cache_path):
                        if file.endswith('.pkl'):
                            os.remove(os.path.join(cache_path, file))
                logger.info("Disk cache cleared")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def cleanup_expired(self, max_age_hours: int = 24) -> int:
        """Limpar itens expirados do cache"""
        removed_count = 0
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        try:
            
            expired_keys = [
                key for key, data in self.embeddings_cache.items()
                if data['timestamp'] < cutoff_time
            ]
            for key in expired_keys:
                del self.embeddings_cache[key]
                removed_count += 1
            
            expired_keys = [
                key for key, data in self.predictions_cache.items()
                if data['timestamp'] < cutoff_time
            ]
            for key in expired_keys:
                del self.predictions_cache[key]
                removed_count += 1
            
            
            for cache_subdir in ["embeddings", "predictions"]:
                cache_path = os.path.join(self.cache_dir, cache_subdir)
                for file in os.listdir(cache_path):
                    if file.endswith('.pkl'):
                        file_path = os.path.join(cache_path, file)
                        if os.path.getmtime(file_path) < cutoff_time:
                            os.remove(file_path)
                            removed_count += 1
            
            logger.info(f"Cache cleanup completed: {removed_count} items removed")
            return removed_count
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            return removed_count


cache_manager = CacheManager()