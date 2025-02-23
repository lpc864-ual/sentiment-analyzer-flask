from flask import Flask, render_template, request, redirect, url_for
from sentiment_model import SentimentAnalyzer
from data.training_data import training_data
from threading import Thread
import time

# Variables globales
app = Flask(__name__)
analyzer = SentimentAnalyzer()
training_in_progress = False

def background_training():
    global training_in_progress
    try:
        # Reentrenar el modelo
        analyzer.train_model()
    finally:
        # Asegurarnos que training_in_progress se establezca en False incluso si hay error
        training_in_progress = False

@app.route('/')
def home():
    global training_in_progress
    if training_in_progress:
        return render_template('training.html')
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']
        sentiment, score = analyzer.analyze_sentiment(text)
        return render_template('result.html', text=text, sentiment=sentiment, score=round(score, 2))

@app.route('/feedback', methods=['POST'])
def feedback():
    global training_in_progress
    
    if request.method == 'POST':
        text = request.form['original_text'].lower()
        feedback_sentiment = request.form['feedback_sentiment']
        
        # Verificar si el texto ya existe en training_data
        text_exists = any(item['text'] == text for item in training_data)
        
        if not text_exists:
            # Añadir solo si no existe
            training_data.append({
                "text": text,
                "sentiment": feedback_sentiment
            })
            
            # Guardar el feedback en el archivo de training_data
            with open('data/training_data.py', 'w', encoding='utf-8') as f:
                f.write('training_data = [\n')
                for item in training_data:
                    f.write(f'    {{"text": "{item["text"]}", "sentiment": "{item["sentiment"]}"}},\n')
                f.write(']\n')
            
            # Iniciar el entrenamiento en background
            training_in_progress = True
            Thread(target=background_training).start()
        
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True, threaded=True)