import os
import nltk
from nltk.corpus import stopwords
import re
import numpy as np
import tensorflow as tf
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from data.training_data import training_data
import pickle

class SentimentAnalyzer:
    def __init__(self):
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('spanish'))
        self.negative_words = {'no', 'ni', 'nunca', 'tampoco', 'nada', 'ningún', 'ninguno', 'nadie'}
        self.stop_words_without_negation = self.stop_words - self.negative_words
        # Critico para evitar que el modelo se confunda con palabras muy similares entre si (p.e. podria memorizar bueno y algun sinonimo de bueno que no sea tan bueno y asigne el valor a dicho valor)
        self.max_words = 5000
        # Critico para mejorar el uso de recursos por parte del modelo y evitar la influencia del ruido
        self.max_len = 20
        
        # Cargar o crear el modelo
        try:
            # Intentar cargar modelo existente
            self.model = tf.keras.models.load_model('model/sentiment_model.h5')
            with open('model/tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            with open('model/label_encoder.pickle', 'rb') as handle:
                self.label_encoder = pickle.load(handle)
        except:
            # Si no existe, entrenar nuevo modelo
            self.train_model()

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        words = text.split()
        words = [w for w in words if w not in self.stop_words_without_negation]
        return ' '.join(words)

    def train_model(self):
        # Preparar datos
        texts = [item['text'] for item in training_data]
        labels = [item['sentiment'] for item in training_data]
        # Preprocesar textos
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Tokenización
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.tokenizer.fit_on_texts(processed_texts)
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        X = pad_sequences(sequences, maxlen=self.max_len)
        
        # Codificar etiquetas
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit_transform(labels)
        y = self.label_encoder.transform(labels)
        print("y: ", y)
        y = tf.keras.utils.to_categorical(y)
        
        # Crear modelo
        self.model = tf.keras.Sequential([
            # 100 representa el tamaño del vector de embedding (representacion vectorial de las palabras)
            tf.keras.layers.Embedding(self.max_words, 100, input_length=self.max_len),
            # 64 representa el numero de neuronas en la capa LSTM: cuanto mayor sea mas recordara pero posiblemente se sobreajuste (aprenda bien el conjunto de datos de entrenamiento pero no generalizara bien)
            tf.keras.layers.LSTM(64, return_sequences=True),
            # Igual que antes pero en otra capa
            # Dichos valores suelen ajustarse si se tiene sobreajuste (diferencia entre el conjunto de datos de entrenamiento y el de validacion)
            tf.keras.layers.LSTM(32),
            # Igual que antes: cuanto mayor mas rapdio pero puede que se sobreajuste y no generalizar bien
            # relu: funcion de  filtrado (p.e. si es 1 deja pasar, sino no)
            tf.keras.layers.Dense(16, activation='relu'),
            # Cuanto mas alto, mas computo y uso de recursos pero busca aprender patrones mas complejos mejorando la generalizacion
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        # Compilar modelo
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Entrenar modelo
        # batch_size: numero de muestras a utilizar. Si varia mucho el accuracy de un lote a otro, bajar el batch_size
        self.model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
        
        # Crear la carpeta 'model' si no existe
        os.makedirs("model", exist_ok=True)

        # Guardar modelo y preprocesadores
        self.model.save('model/sentiment_model.h5')
        with open('model/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle)
        with open('model/label_encoder.pickle', 'wb') as handle:
            pickle.dump(self.label_encoder, handle)

    def analyze_sentiment(self, text):
        # Preprocesar texto
        processed_text = self.preprocess_text(text)
        
        # Tokenizar y padear
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded = pad_sequences(sequence, maxlen=self.max_len)
        
        # Predecir
        prediction = self.model.predict(padded)[0] # 
        sentiment_idx = np.argmax(prediction)
        print(sentiment_idx)
        sentiment = self.label_encoder.inverse_transform([sentiment_idx])[0]
        score = float(prediction[sentiment_idx])
        
        return sentiment, score