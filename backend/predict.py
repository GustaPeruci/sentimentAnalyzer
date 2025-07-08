import os
import pickle
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class SentimentPredictor:
    def __init__(self, model_path="models/sentiment_model.pkl"):
        self.device = 'cpu'
        self.bert_model = None
        self.classifier = None
        self.classes = None
        self.is_loaded = False
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.bert_model = SentenceTransformer('models/all-MiniLM-L6-v2', device=self.device)
            except Exception as e:
                print(e)
                self.bert_model = None
        
        if os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path="models/sentiment_model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.classes = model_data['classes']
        self.is_loaded = True
        print(f"Model loaded from {model_path}")
    
    def predict_sentiment(self, text):
        if not text or not text.strip():
            return {
                'error': 'Empty text provided',
                'sentiment': None,
                'confidence': 0.0,
                'probabilities': {}
            }
        
        try:
            if TRANSFORMERS_AVAILABLE and self.is_loaded and self.bert_model and self.classifier:
                embedding = self.bert_model.encode(
                    [text.strip()], 
                    device=self.device, 
                    convert_to_numpy=True
                )
                
                prediction = self.classifier.predict(embedding)[0]
                probabilities = self.classifier.predict_proba(embedding)[0]
                
                confidence = float(np.max(probabilities))
                
                prob_dict = {
                    str(class_name): float(prob) 
                    for class_name, prob in zip(self.classes, probabilities)
                }
                
                sentiment_key = prediction
            else:
                sentiment_key, confidence, prob_dict = self._analyze_sentiment_advanced(text.strip())
            
            display_mapping = {
                'alegria': 'Alegria',
                'tristeza': 'Tristeza', 
                'surpresa': 'Surpresa',
                'raiva': 'Raiva',
                'positive': 'Alegria',
                'negative': 'Tristeza',
                'neutral': 'Surpresa'
            }
            
            display_sentiment = display_mapping.get(sentiment_key, sentiment_key.title())
            
            return {
                'sentiment': display_sentiment,
                'sentiment_key': sentiment_key,
                'confidence': confidence,
                'probabilities': prob_dict,
                'text': text.strip()
            }
            
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'sentiment': None,
                'confidence': 0.0,
                'probabilities': {}
            }
    
    def _simple_sentiment_analysis(self, text):
        return self._analyze_sentiment_advanced(text)
    
    def _analyze_sentiment_advanced(self, text):
        if not text or not text.strip():
            return 'surpresa', 0.5, {'alegria': 0.25, 'tristeza': 0.25, 'raiva': 0.25, 'surpresa': 0.25}
        
        processed_text = self._preprocess_text(text)
        
        pattern_scores = self._analyze_patterns(processed_text)
        lexicon_scores = self._analyze_lexicons(processed_text)
        context_scores = self._analyze_context(processed_text)
        
        final_scores = {'alegria': 0, 'tristeza': 0, 'raiva': 0, 'surpresa': 0}
        
        for sentiment in final_scores:
            final_scores[sentiment] = (
                pattern_scores[sentiment] * 1.5 +  
                lexicon_scores[sentiment] * 1.2 +  
                context_scores[sentiment] * 1.0   
            )
        
        surprise_expectation_patterns = [
            'não_esperava que',  
            'não_esperava',
            'nunca imaginei que',
            'não_imaginava que',
            'surpreendeu',
            'surpresa'
        ]
        
        positive_outcomes = [
            'incrível', 'excelente', 'fantástico', 'maravilhoso', 
            'qualidade', 'ótimo', 'perfeito', 'tão bom'
        ]

        has_surprise_expectation = any(pattern in processed_text for pattern in surprise_expectation_patterns)
        has_positive_outcome = any(outcome in processed_text for outcome in positive_outcomes)
        
        if has_surprise_expectation and has_positive_outcome:
            final_scores['surpresa'] += 8.0
            final_scores['alegria'] *= 0.3
            final_scores['tristeza'] *= 0.3
            final_scores['raiva'] *= 0.3
        elif has_surprise_expectation:
            final_scores['surpresa'] += 5.0
            final_scores['alegria'] *= 0.5
        
        anger_phrases = [
            'muito insatisfeito',
            'estou muito insatisfeito',
            'não_gostei nada',
            'péssimo produto',
            'horrível produto',
            'totalmente insatisfeito'
        ]
        
        has_strong_anger = any(phrase in processed_text for phrase in anger_phrases)
        
        anger_indicators = [
            'insatisfeito' in processed_text,
            'não_gostei' in processed_text,
            'péssimo' in processed_text,
            'horrível' in processed_text,
            'terrível' in processed_text,
            'detesto' in processed_text,
            'odeio' in processed_text,
            'muito' in processed_text and any(neg in processed_text for neg in ['insatisfeito', 'ruim', 'péssimo'])
        ]
        
        anger_count = sum(anger_indicators)
        
        if has_strong_anger or anger_count >= 2:
            final_scores['raiva'] += 8.0
            final_scores['tristeza'] *= 0.4
        elif anger_count >= 1 and 'insatisfeito' in processed_text:
            final_scores['raiva'] += 4.0
            final_scores['tristeza'] *= 0.6
        
        joy_phrases = [
            'muito satisfeito',
            'estou muito satisfeito',
            'adorei',
            'excelente produto',
            'perfeito',
            'funcionou perfeitamente'
        ]
        
        has_strong_joy = any(phrase in processed_text for phrase in joy_phrases)
        
        joy_indicators = [
            'satisfeito' in processed_text,
            'gostei' in processed_text,
            'ótimo' in processed_text,
            'excelente' in processed_text,
            'perfeito' in processed_text,
            'funcionou' in processed_text and 'bem' in processed_text
        ]
        
        joy_count = sum(joy_indicators)
        
        if has_strong_joy or (joy_count >= 2 and not has_surprise_expectation):
            final_scores['alegria'] += 6.0
        elif joy_count >= 1 and not has_surprise_expectation:
            final_scores['alegria'] += 3.0
        
        max_sentiment = max(final_scores.keys(), key=lambda k: final_scores[k])
        max_score = final_scores[max_sentiment]
        
        total_score = sum(final_scores.values())
        if total_score > 0:
            confidence = max_score / total_score
            if max_score > 5:
                confidence = min(confidence * 1.3, 0.95)
            elif max_score > 3:
                confidence = min(confidence * 1.2, 0.90)
        else:
            max_sentiment = 'surpresa'
            confidence = 0.5
        
        confidence = max(confidence, 0.6)
        
        if total_score > 0:
            probabilities = {k: v / total_score for k, v in final_scores.items()}
        else:
            probabilities = {'alegria': 0.25, 'tristeza': 0.25, 'raiva': 0.25, 'surpresa': 0.25}
        
        return max_sentiment, confidence, probabilities
    
    def _preprocess_text(self, text):
        """Preprocess text for analysis"""
        import re
        text = text.lower().strip()
        
        text = re.sub(r'\s+', ' ', text)
        
        text = re.sub(r'\bnão\s+', 'não_', text)
        
        return text
    
    def _analyze_patterns(self, text):
        """Analyze text using regex patterns"""
        import re
        scores = {'alegria': 0.0, 'tristeza': 0.0, 'raiva': 0.0, 'surpresa': 0.0}
        
        positive_patterns = [
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
        
        negative_patterns = [
            r'não funciona',
            r'parou de funcionar',
            r'quebrou depois',
            r'não vale a pena',
            r'péssimo (produto|serviço)',
            r'muito ruim',
            r'horrível (qualidade|produto)',
            r'decepção total',
            r'não recomendo',
            r'não gostei',
            r'não gosto',
            r'muito insatisfeito',
            r'estou insatisfeito',
            r'estou muito insatisfeito',
            r'totalmente insatisfeito',
            r'completamente insatisfeito',
            r'extremamente insatisfeito'
        ]
        
        anger_patterns = [
            r'é um absurdo',
            r'totalmente inaceitável',
            r'que ridículo',
            r'uma vergonha',
            r'péssimo atendimento',
            r'revoltante',
            r'estou muito insatisfeito.*não gostei',
            r'muito insatisfeito.*produto.*não gostei',
            r'insatisfeito.*produto.*não gostei',
            r'não gostei.*produto.*insatisfeito',
            r'muito insatisfeito.*não gostei',
            r'totalmente insatisfeito.*não gostei',
            r'completamente insatisfeito.*não gostei',
            r'extremamente insatisfeito.*não gostei'
        ]
        
        surprise_patterns = [
            r'não esperava (que|isso)',
            r'surpreendeu positivamente',
            r'surpreendeu negativamente',
            r'bem diferente do esperado',
            r'muito melhor que esperava',
            r'pior que esperava'
        ]
        
        # Check patterns
        for pattern in positive_patterns:
            if re.search(pattern, text):
                scores['alegria'] += 2.5
        
        for pattern in negative_patterns:
            if re.search(pattern, text):
                scores['tristeza'] += 2.5
        
        for pattern in anger_patterns:
            if re.search(pattern, text):
                scores['raiva'] += 2.5
        
        for pattern in surprise_patterns:
            if re.search(pattern, text):
                scores['surpresa'] += 2.5
        
        return scores
    
    def _analyze_lexicons(self, text):
        scores = {'alegria': 0.0, 'tristeza': 0.0, 'raiva': 0.0, 'surpresa': 0.0}
        
        # Positive lexicon with weights
        positive_lexicon = {
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
        
        negative_lexicon = {
            'péssimo': 3, 'horrível': 3, 'terrível': 3, 'inaceitável': 3,
            'lamentável': 3, 'vergonhoso': 3, 'desastroso': 3,
            'não funciona': 3, 'quebrou': 3, 'defeituoso': 3,
            'insatisfeito': 3, 'muito insatisfeito': 3, 'insatisfatório': 3,
            'ruim': 2, 'problema': 2, 'defeito': 2, 'atrasada': 2,
            'parou': 2, 'decepcionado': 2, 'frustrado': 2, 'triste': 2,
            'não gostei': 3, 'não gosto': 3, 'detestei': 3, 'odeio': 3,
            'decepção': 2, 'pior': 2, 'falhou': 2, 'desapontado': 2,
            'fraco': 1, 'lento': 1, 'demorado': 1, 'difícil': 1,
            'complicado': 1, 'confuso': 1, 'limitado': 1
        }
        
        anger_lexicon = {
            'raiva': 3, 'ódio': 3, 'furioso': 3, 'irritado': 2,
            'absurdo': 3, 'ridículo': 3, 'inadmissível': 3,
            'revoltante': 3, 'indignado': 2, 'chateado': 2,
            'bravo': 2, 'nervoso': 2, 'incomodado': 1,
            'muito insatisfeito': 3, 'totalmente insatisfeito': 3,
            'extremamente insatisfeito': 3, 'completamente insatisfeito': 3,
            'não gostei nada': 3, 'detesto': 3, 'odeio': 3,
            'que lixo': 3, 'uma porcaria': 3, 'um lixo': 3
        }
        
        surprise_lexicon = {
            'surpresa': 2, 'inesperado': 2, 'não esperava': 3,
            'surpreendente': 2, 'impressionante': 2, 'inacreditável': 3,
            'espantoso': 2, 'chocante': 3, 'uau': 2, 'wow': 2,
            'nossa': 1, 'caramba': 1
        }
        
        lexicons = {
            'alegria': positive_lexicon,
            'tristeza': negative_lexicon,
            'raiva': anger_lexicon,
            'surpresa': surprise_lexicon
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
    
    def _analyze_context(self, text):
        scores = {'alegria': 0.0, 'tristeza': 0.0, 'raiva': 0.0, 'surpresa': 0.0}
        
        if 'produto' in text:
            if any(word in text for word in ['funcionou', 'funciona', 'bem', 'perfeitamente']):
                scores['alegria'] += 2
            elif any(word in text for word in ['não funciona', 'quebrou', 'defeito']):
                scores['tristeza'] += 2
        
        if any(period in text for period in ['dias', 'semanas', 'meses']):
            if 'atrasada' in text or 'atraso' in text:
                scores['tristeza'] += 1.5
                scores['raiva'] += 1
        
        if 'qualidade' in text:
            if any(word in text for word in ['boa', 'ótima', 'excelente']):
                scores['alegria'] += 1.5
            elif any(word in text for word in ['ruim', 'péssima', 'baixa']):
                scores['tristeza'] += 1.5
        
        if 'recomendo' in text:
            scores['alegria'] += 2
        elif 'não recomendo' in text:
            scores['tristeza'] += 2
        
        return scores
    
    def batch_predict(self, texts):
        if not self.is_loaded:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        results = []
        for text in texts:
            result = self.predict_sentiment(text)
            results.append(result)
        
        return results

predictor = SentimentPredictor()

def predict_sentiment(text):
    return predictor.predict_sentiment(text)

def get_model_info():
    if not predictor.is_loaded:
        return {
            'loaded': False,
            'message': 'Using keyword-based sentiment analysis',
            'mode': 'keyword-based'
        }
    
    return {
        'loaded': True,
        'classes': predictor.classes.tolist() if predictor.classes is not None else ['alegria', 'tristeza', 'raiva', 'surpresa'],
        'device': predictor.device,
        'mode': 'bert-based' if TRANSFORMERS_AVAILABLE else 'keyword-based'
    }

def test_predictions():
    test_texts = [
        "Este produto é incrível, superou minhas expectativas!",
        "A entrega está 3 dias atrasada...",
        "Esse produto parou de funcionar depois de uma semana",
        "Eu não esperava essa funcionalidade..."
    ]
    
    for text in test_texts:
        result = predict_sentiment(text)
        print(f"\nTexto: '{text}'")
        print(f"Sentimento: {result.get('sentiment', 'N/A')}")
        print(f"Confiança: {result.get('confidence', 0):.4f}")

if __name__ == "__main__":
    test_predictions()
