import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    # Remove caracteres não alfanuméricos e converte para minúsculas
    if not isinstance(text, str):
        text = ''  # Convert non-string type (like NaN) to empty string
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I | re.A)
    text = text.lower()
    text = text.strip()

    # Tokenização e remoção de stopwords
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)
