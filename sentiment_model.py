import nltk
from nltk.corpus import stopwords
import re
from textblob import TextBlob


# Clase para análisis de sentimiento
class SentimentAnalyzer:
    # Constructor
    def __init__(self):
        nltk.download('stopwords')
        # Descarga recursos necesarios de NLTK
        self.stop_words = set(stopwords.words('spanish'))

    # Método para preprocesar texto
    def preprocess_text(self, text):
        # Convertir a minúsculas
        text = text.lower()
        # Eliminar caracteres especiales
        text = re.sub(r'[^\w\s]', '', text)
        # Eliminar números
        text = re.sub(r'\d+', '', text)
        # Eliminar stopwords
        words = text.split()
        words = [w for w in words if w not in self.stop_words]
        return ' '.join(words)

    def analyze_sentiment(self, text):
        # Preprocesar el texto
        processed_text = self.preprocess_text(text)
        
        # Realizar análisis de sentimiento
        analysis = TextBlob(processed_text)
        
        # Determinar el sentimiento
        if analysis.sentiment.polarity > 0:
            return 'positivo', analysis.sentiment.polarity
        elif analysis.sentiment.polarity < 0:
            return 'negativo', analysis.sentiment.polarity
        else:
            return 'neutral', analysis.sentiment.polarity