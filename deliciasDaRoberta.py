import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
import torch  # Import torch

# Load data
data = pd.read_csv('train.csv')

# Fill missing values
data['text'] = data['text'].fillna(" ")

# Initialize tokenizer and RoBERTa model pre-trained
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Create a classification pipeline
nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Function to predict fake news
def predict_fake_news(text):
    # Use the pipeline for prediction, handling text truncation automatically
    prediction = nlp_pipeline(text, truncation=True)
    return prediction[0]['label']

# Apply prediction function on dataframe
data['predicted_label'] = data['text'].apply(predict_fake_news)

# Convert predicted labels back to binary form (0 or 1)
data['predicted_label'] = data['predicted_label'].replace({'LABEL_0': 0, 'LABEL_1': 1})

# Optionally print or save the results
print(data[['id', 'title', 'predicted_label']])
