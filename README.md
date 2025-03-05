# Spanish Sentiment Analyzer

A deep learning-based sentiment analyzer for Spanish text. The application analyzes texts and classifies their sentiment as positive, neutral, or negative.

## Features

- Spanish language sentiment analysis
- Intuitive web interface
- Feedback system for model improvement
- Automatic retraining with new data
- Natural Language Processing with NLTK
- LSTM neural network using TensorFlow/Keras

## Requirements

- Python 3.10 (required for compatibility with dependencies)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/[username]/spanish-sentiment-analyzer.git
cd spanish-sentiment-analyzer
```

2. Create a virtual environment with Python 3.10 (required):
```bash
# On Windows:
py -V:3.10 -m venv venv
venv\Scripts\activate

# On macOS/Linux:
python3.10 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
py app.py
```

2. Open your browser and visit `http://localhost:5000`

3. Enter the text you want to analyze and click "Analyze"

4. Provide feedback to help improve the model

## Project Structure

```
.
├── app.py                 # Main Flask application
├── sentiment_model.py     # Sentiment analysis model
├── data/
│   └── training_data.py   # Training data
├── model/                 # Saved model files
├── static/
│   └── style.css         # CSS styles
├── templates/            # HTML templates
│   ├── index.html
│   ├── result.html
│   └── training.html
├── requirements.txt      # Project dependencies
├── LICENSE.md           # License information
└── README.md            # This file
```

## Model

The system uses an LSTM (Long Short-Term Memory) neural network architecture implemented with TensorFlow/Keras. Key features:

- Text preprocessing with NLTK
- Spanish negation handling
- 100-dimensional word embeddings
- LSTM layers with 64 and 32 units
- Dense layer with ReLU activation
- Dropout for overfitting prevention

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes and commit (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Main Dependencies

- Flask
- TensorFlow
- NLTK
- NumPy
- Pandas
- scikit-learn

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Thanks to all contributors who have helped improve the model
- Special thanks to the Spanish NLP community for providing resources and guidance
- Built with Flask and TensorFlow

## Contact

For questions or suggestions, please open an issue in the GitHub repository.