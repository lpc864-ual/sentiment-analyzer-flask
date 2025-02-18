# Sentiment Analyzer Flask

A web application built with Flask that determines whether an opinion in English is positive, negative, or neutral.

## Description

This project implements a web application that allows users to input text in English and receive sentiment analysis results. The application uses NLTK and TextBlob for natural language processing and Flask for the web interface.

## Features

- English text sentiment analysis
- Intuitive web interface
- Text preprocessing (removal of special characters, numbers, and stopwords)
- Results visualization with sentiment score

## Technologies Used

- Python 3.12.2
- Flask
- NLTK
- TextBlob
- HTML/CSS

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sentiment-analyzer-flask.git
cd sentiment-analyzer-flask
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Download required NLTK resources:
```python
python
>>> import nltk
>>> nltk.download('stopwords')
>>> exit()
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open a web browser and visit:
```
http://localhost:5000
```

3. Enter the text to analyze and click "Analyze"

## Project Structure

```
sentiment_analyzer/
│
├── static/
│   └── style.css           # Application styles
│
├── templates/
│   ├── index.html          # Main page
│   └── result.html         # Results page
│
├── app.py                  # Flask application
├── sentiment_model.py      # Analysis model
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Known Limitations

- The analysis is designed for English text
- The development server should not be used in production
- Basic sentiment analysis model (TextBlob) has limited accuracy

## Planned Future Improvements

- Implement more advanced sentiment analysis models
- Add batch analysis support
- Improve accuracy using modern machine learning models
- Add graphical visualizations of results
- Implement caching for better performance

## Contributing

Contributions are welcome. To contribute:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive license that allows you to:
- ✔️ Use the code commercially
- ✔️ Modify the code
- ✔️ Distribute the code
- ✔️ Use the code privately
- ✔️ Sublicense the code

The only requirement is that you include the original copyright and license notice in any copy of the code.

## Contact

Your Name - email@example.com

Project Link: [https://github.com/yourusername/sentiment-analyzer-flask](https://github.com/yourusername/sentiment-analyzer-flask)

## Acknowledgments

- [NLTK](https://www.nltk.org/)
- [TextBlob](https://textblob.readthedocs.io/)
- [Flask](https://flask.palletsprojects.com/)
