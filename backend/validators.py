"""
Validadores para entrada de dados do Zonalyze
Garante integridade e segurança dos dados
"""
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Resultado de uma validação"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    cleaned_data: Optional[str] = None

class TextValidator:
    """Validador para textos de análise de sentimentos"""
    
    # Configurações
    MIN_LENGTH = 3
    MAX_LENGTH = 5000
    
    # Padrões suspeitos
    SPAM_PATTERNS = [
        r'(.)\1{10,}',  # Repetição excessiva de caracteres
        r'[A-Z]{20,}',  # Texto em maiúsculas excessivo
        r'(https?://\S+){3,}',  # Múltiplos links
    ]
    
    # Palavras proibidas básicas (pode ser expandido)
    BLOCKED_WORDS = [
        'spam', 'test' * 10,  # Palavras óbvias de teste
    ]
    
    @classmethod
    def validate_text(cls, text: str) -> ValidationResult:
        """
        Valida um texto para análise de sentimentos
        
        Args:
            text: Texto a ser validado
            
        Returns:
            ValidationResult com resultado da validação
        """
        errors = []
        warnings = []
        
        # Verificar se texto existe
        if not text or not isinstance(text, str):
            errors.append("Texto é obrigatório e deve ser uma string")
            return ValidationResult(False, errors, warnings)
        
        # Limpar texto básico
        cleaned_text = text.strip()
        
        # Verificar comprimento
        if len(cleaned_text) < cls.MIN_LENGTH:
            errors.append(f"Texto muito curto. Mínimo: {cls.MIN_LENGTH} caracteres")
        
        if len(cleaned_text) > cls.MAX_LENGTH:
            errors.append(f"Texto muito longo. Máximo: {cls.MAX_LENGTH} caracteres")
        
        # Verificar padrões de spam
        for pattern in cls.SPAM_PATTERNS:
            if re.search(pattern, cleaned_text):
                warnings.append("Texto pode conter padrões suspeitos")
                break
        
        # Verificar palavras bloqueadas
        text_lower = cleaned_text.lower()
        for word in cls.BLOCKED_WORDS:
            if word in text_lower:
                warnings.append(f"Texto contém palavra suspeita: '{word}'")
        
        # Verificar se é principalmente numérico
        if re.sub(r'[^a-zA-ZÀ-ÿ]', '', cleaned_text) == '':
            warnings.append("Texto contém apenas números/símbolos")
        
        # Verificar encoding válido
        try:
            cleaned_text.encode('utf-8')
        except UnicodeEncodeError:
            errors.append("Texto contém caracteres inválidos")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            cleaned_data=cleaned_text if is_valid else None
        )

class APIValidator:
    """Validador para requests da API"""
    
    @staticmethod
    def validate_predict_request(data: dict) -> ValidationResult:
        """Valida request do endpoint /predict"""
        errors = []
        warnings = []
        
        # Verificar estrutura básica
        if not isinstance(data, dict):
            errors.append("Request deve ser um objeto JSON")
            return ValidationResult(False, errors, warnings)
        
        # Verificar campo 'text'
        if 'text' not in data:
            errors.append("Campo 'text' é obrigatório")
            return ValidationResult(False, errors, warnings)
        
        # Validar o texto
        text_validation = TextValidator.validate_text(data['text'])
        errors.extend(text_validation.errors)
        warnings.extend(text_validation.warnings)
        
        # Verificar campos extras não esperados
        expected_fields = {'text', 'options'}  # 'options' para futuras configurações
        extra_fields = set(data.keys()) - expected_fields
        if extra_fields:
            warnings.append(f"Campos extras ignorados: {', '.join(extra_fields)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data=text_validation.cleaned_data
        )
    
    @staticmethod
    def validate_batch_request(data: dict) -> ValidationResult:
        """Valida request do endpoint /batch-predict"""
        errors = []
        warnings = []
        
        if not isinstance(data, dict):
            errors.append("Request deve ser um objeto JSON")
            return ValidationResult(False, errors, warnings)
        
        if 'texts' not in data:
            errors.append("Campo 'texts' é obrigatório")
            return ValidationResult(False, errors, warnings)
        
        texts = data['texts']
        if not isinstance(texts, list):
            errors.append("Campo 'texts' deve ser uma lista")
            return ValidationResult(False, errors, warnings)
        
        if len(texts) == 0:
            errors.append("Lista de textos não pode estar vazia")
            return ValidationResult(False, errors, warnings)
        
        if len(texts) > 100:  # Limite para batch processing
            errors.append("Máximo 100 textos por batch")
            return ValidationResult(False, errors, warnings)
        
        # Validar cada texto
        cleaned_texts = []
        for i, text in enumerate(texts):
            text_validation = TextValidator.validate_text(text)
            if not text_validation.is_valid:
                errors.append(f"Texto {i+1}: {'; '.join(text_validation.errors)}")
            else:
                cleaned_texts.append(text_validation.cleaned_data)
            
            if text_validation.warnings:
                warnings.append(f"Texto {i+1}: {'; '.join(text_validation.warnings)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data=cleaned_texts if len(errors) == 0 else None
        )