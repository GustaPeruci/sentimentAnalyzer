import re
import json
from typing import Dict, List, Tuple

class AdvancedSentimentAnalyzer:
    def __init__(self):
        self.initialize_lexicons()
        self.initialize_patterns()
        
    def initialize_lexicons(self):
        
        self.positive_lexicon = {
            
            'excelente': 3, 'perfeito': 3, 'maravilhoso': 3, 'fantástico': 3,
            'incrível': 3, 'espetacular': 3, 'excepcional': 3, 'brilhante': 3,
            'superou': 3, 'perfeitamente': 3, 'extraordinário': 3,
            
            
            'ótimo': 2, 'bom': 2, 'muito bom': 2, 'satisfeito': 2,
            'feliz': 2, 'gostei': 2, 'adorei': 2, 'recomendo': 2,
            'melhor': 2, 'funcionou': 2, 'funciona': 2, 'eficiente': 2,
            'bem': 2, 'funcional': 2, 'útil': 2, 'qualidade': 2,
            
            
            'legal': 1, 'bacana': 1, 'interessante': 1, 'ok': 1,
            'okay': 1, 'satisfatório': 1, 'adequado': 1, 'aceitável': 1
        }
        
        
        self.negative_lexicon = {
            
            'péssimo': 3, 'horrível': 3, 'terrível': 3, 'inaceitável': 3,
            'lamentável': 3, 'vergonhoso': 3, 'desastroso': 3,
            'não funciona': 3, 'quebrou': 3, 'defeituoso': 3,
            
            
            'ruim': 2, 'problema': 2, 'defeito': 2, 'atrasada': 2,
            'parou': 2, 'decepcionado': 2, 'frustrado': 2, 'triste': 2,
            'não gostei': 2, 'decepção': 2, 'pior': 2, 'falhou': 2,
            
            
            'fraco': 1, 'lento': 1, 'demorado': 1, 'difícil': 1,
            'complicado': 1, 'confuso': 1, 'limitado': 1
        }
        
        
        self.anger_lexicon = {
            'raiva': 3, 'ódio': 3, 'furioso': 3, 'irritado': 2,
            'absurdo': 3, 'ridículo': 3, 'inadmissível': 3,
            'revoltante': 3, 'indignado': 2, 'chateado': 2,
            'bravo': 2, 'nervoso': 2, 'incomodado': 1
        }
        
        
        self.surprise_lexicon = {
            'surpresa': 2, 'inesperado': 2, 'não esperava': 4,
            'surpreendente': 2, 'impressionante': 2, 'inacreditável': 3,
            'espantoso': 2, 'chocante': 3, 'uau': 2, 'wow': 2,
            'nossa': 1, 'caramba': 1, 'surpreendeu': 3, 'surpreendi': 3,
            'nunca imaginei': 4, 'não imaginava': 4, 'não sabia': 2,
            'diferente do esperado': 3, 'além do esperado': 3
        }
    
    def initialize_patterns(self):
        """Initialize regex patterns for complex sentiment detection"""
        self.positive_patterns = [
            r'funcionou (muito )?bem',
            r'funciona (muito )?bem',
            r'perfeitamente bem',
            r'funcionou perfeitamente',
            r'superou (as )?expectativas',
            r'muito (bom|boa|satisfeito)',
            r'recomendo (muito|bastante)?',
            r'vale a pena',
            r'custo benefício',
            r'melhor que esperava'
        ]
        
        self.negative_patterns = [
            r'não funciona',
            r'parou de funcionar',
            r'quebrou depois',
            r'não vale a pena',
            r'péssimo (produto|serviço)',
            r'muito ruim',
            r'horrível (qualidade|produto)',
            r'decepção total',
            r'não recomendo'
        ]
        
        self.anger_patterns = [
            r'é um absurdo',
            r'totalmente inaceitável',
            r'que ridículo',
            r'uma vergonha',
            r'péssimo atendimento',
            r'revoltante'
        ]
        
        self.surprise_patterns = [
            r'não esperava (que|isso)',
            r'não esperava que .* (tivesse|fosse|seria)',
            r'surpreendeu positivamente',
            r'surpreendeu negativamente',
            r'bem diferente do esperado',
            r'muito melhor que esperava',
            r'pior que esperava',
            r'nunca imaginei que',
            r'não imaginava que',
            r'além das expectativas',
            r'superou expectativas',
            r'contrário ao esperado',
            r'diferente do que pensava'
        ]
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        
        text = text.lower().strip()
        
        
        text = re.sub(r'\s+', ' ', text)
        
        
        text = re.sub(r'\bnão\s+', 'não_', text)
        
        return text
    
    def analyze_pattern_matches(self, text: str) -> Dict[str, float]:
        """Analyze text using regex patterns"""
        scores = {'alegria': 0, 'tristeza': 0, 'raiva': 0, 'surpresa': 0}
        
        
        for pattern in self.positive_patterns:
            if re.search(pattern, text):
                scores['alegria'] += 2.5
        
        
        for pattern in self.negative_patterns:
            if re.search(pattern, text):
                scores['tristeza'] += 2.5
        
        
        for pattern in self.anger_patterns:
            if re.search(pattern, text):
                scores['raiva'] += 2.5
        
        
        for pattern in self.surprise_patterns:
            if re.search(pattern, text):
                scores['surpresa'] += 2.5
        
        return scores
    
    def analyze_lexicon_matches(self, text: str) -> Dict[str, float]:
        """Analyze text using sentiment lexicons"""
        scores = {'alegria': 0, 'tristeza': 0, 'raiva': 0, 'surpresa': 0}
        
        
        lexicons = {
            'alegria': self.positive_lexicon,
            'tristeza': self.negative_lexicon,
            'raiva': self.anger_lexicon,
            'surpresa': self.surprise_lexicon
        }
        
        for sentiment, lexicon in lexicons.items():
            for word, weight in lexicon.items():
                if word in text:
                    
                    if f'não_{word}' in text or f'não {word}' in text:
                        
                        if sentiment == 'alegria':
                            scores['tristeza'] += weight * 0.8
                        elif sentiment == 'tristeza':
                            scores['alegria'] += weight * 0.8
                    else:
                        scores[sentiment] += weight
        
        return scores
    
    def analyze_context(self, text: str) -> Dict[str, float]:
        """Analyze contextual clues in the text"""
        scores = {'alegria': 0, 'tristeza': 0, 'raiva': 0, 'surpresa': 0}
        
        
        surprise_indicators = ['não esperava', 'nunca imaginei', 'não imaginava', 'surpreendeu']
        positive_indicators = ['incrível', 'excelente', 'ótima', 'fantástico', 'maravilhoso', 'qualidade']
        
        has_surprise_context = any(indicator in text for indicator in surprise_indicators)
        has_positive_outcome = any(indicator in text for indicator in positive_indicators)
        
        if has_surprise_context and has_positive_outcome:
            
            scores['surpresa'] += 5.0
            
            scores['alegria'] = max(0, scores['alegria'] - 2.0)
        elif has_surprise_context:
            scores['surpresa'] += 3.0
        
        
        if 'produto' in text:
            if any(word in text for word in ['funcionou', 'funciona', 'bem', 'perfeitamente']):
                if not has_surprise_context:  
                    scores['alegria'] += 2
            elif any(word in text for word in ['não funciona', 'quebrou', 'defeito']):
                scores['tristeza'] += 2
        
        
        if any(period in text for period in ['dias', 'semanas', 'meses']):
            if 'atrasada' in text or 'atraso' in text:
                scores['tristeza'] += 1.5
                scores['raiva'] += 1
        
        
        if 'qualidade' in text:
            if any(word in text for word in ['boa', 'ótima', 'excelente']):
                if not has_surprise_context:  
                    scores['alegria'] += 1.5
            elif any(word in text for word in ['ruim', 'péssima', 'baixa']):
                scores['tristeza'] += 1.5
        
        
        if 'preço' in text and ('incrível' in text or 'qualidade' in text):
            if has_surprise_context:
                scores['surpresa'] += 2.0
        
        
        if 'recomendo' in text:
            if not has_surprise_context:
                scores['alegria'] += 2
        elif 'não recomendo' in text:
            scores['tristeza'] += 2
        
        return scores
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Perform comprehensive sentiment analysis
        
        Returns:
            Tuple of (sentiment, confidence, probabilities)
        """
        if not text or not text.strip():
            return 'surpresa', 0.5, {'alegria': 0.25, 'tristeza': 0.25, 'raiva': 0.25, 'surpresa': 0.25}
        
        
        processed_text = self.preprocess_text(text)
        
        
        pattern_scores = self.analyze_pattern_matches(processed_text)
        lexicon_scores = self.analyze_lexicon_matches(processed_text)
        context_scores = self.analyze_context(processed_text)
        
        
        final_scores = {'alegria': 0, 'tristeza': 0, 'raiva': 0, 'surpresa': 0}
        
        for sentiment in final_scores:
            final_scores[sentiment] = (
                pattern_scores[sentiment] * 1.5 +  
                lexicon_scores[sentiment] * 1.2 +  
                context_scores[sentiment] * 1.0    
            )
        
        
        max_sentiment = max(final_scores.keys(), key=lambda k: final_scores[k])
        max_score = final_scores[max_sentiment]
        
        
        total_score = sum(final_scores.values())
        if total_score > 0:
            confidence = max_score / total_score
            
            if max_score > 3:
                confidence = min(confidence * 1.2, 0.95)
        else:
            
            max_sentiment = 'surpresa'
            confidence = 0.5
        
        
        confidence = max(confidence, 0.6)
        
        
        if total_score > 0:
            probabilities = {k: v / total_score for k, v in final_scores.items()}
        else:
            probabilities = {'alegria': 0.25, 'tristeza': 0.25, 'raiva': 0.25, 'surpresa': 0.25}
        
        return max_sentiment, confidence, probabilities


if __name__ == "__main__":
    analyzer = AdvancedSentimentAnalyzer()
    
    test_phrases = [
        "o produto funcionou perfeitamente bem",
        "péssimo produto, não recomendo",
        "é um absurdo esse atraso",
        "não esperava que fosse tão bom",
        "muito satisfeito com a compra"
    ]
    
    for phrase in test_phrases:
        sentiment, confidence, probs = analyzer.analyze_sentiment(phrase)
        print(f"'{phrase}' -> {sentiment} ({confidence:.2f})")