import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Carregar dados
data = pd.read_csv('train.csv')

# Preencher valores faltantes
data['text'] = data['text'].fillna(" ")

# Inicializar o tokenizer e o modelo RoBERTa pré-treinado
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Criar uma pipeline de classificação
nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Função para fazer a predição
def predict_fake_news(text):
    return nlp_pipeline(text)[0]['label']

# Aplicar a função de predição no dataframe
data['predicted_label'] = data['text'].apply(predict_fake_news)

# Converter rótulos preditos de volta para forma binária (0 ou 1)
data['predicted_label'] = data['predicted_label'].replace({'LABEL_0': 0, 'LABEL_1': 1})

# Se tivermos etiquetas reais, avaliar o modelo
if 'label' in data.columns:
    print(classification_report(data['label'], data['predicted_label']))

# Salvar ou imprimir os resultados
print(data[['id', 'title', 'predicted_label']])
