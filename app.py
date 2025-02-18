from flask import Flask, render_template, request
from sentiment_model import SentimentAnalyzer

# Variables globales: inicializar Flask para encontrar recursos en el modulo actual y objeto SentimentAnalyzer
app = Flask(__name__)
analyzer = SentimentAnalyzer()

# Definimos ruta principal
@app.route('/')
def home():
    return render_template('index.html')

# Definimos ruta para análisis de sentimiento
@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']
        sentiment, score = analyzer.analyze_sentiment(text)
        return render_template('result.html', text=text, sentiment=sentiment, score=round(score, 2))

# Ejecutar la aplicación ante la ejecución del script
if __name__ == '__main__':
    # Habilitar modo debug para reiniciar la aplicación automáticamente al guardar cambios
    app.run(debug=True)