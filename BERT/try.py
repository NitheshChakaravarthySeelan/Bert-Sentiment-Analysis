import torch
import streamlit as st
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from bert import BertModelForSequenceClassification
from datasets import load_dataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)

dataset = load_dataset("imdb")
tokenized_datasets = dataset.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True, max_length=64), batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


vocab_size = tokenizer.vocab_size
d_model = 512
h = 8
num_layers = 8
d_ff = 2048
seq_len = 64
num_classes = 2
dropout = 0.1

model = BertModelForSequenceClassification(vocab_size, d_model, h, num_layers, d_ff, seq_len, num_classes, dropout)
model.load_state_dict(torch.load('bert_imdb_sentiment_model.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_sentiment(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs, dim=1)
    
    return "Positive" if predicted.item() == 1 else "Negative"

def main():
    st.title("Sentiment Analysis with BERT")
    
    st.write("Enter a text below to get its sentiment classification.")
    
    user_input = st.text_area("Input Text", "Type a sentence or paragraph here...")
    
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            sentiment = predict_sentiment(user_input)
            st.write(f"Sentiment: **{sentiment}**")
        else:
            st.write("Please enter some text to analyze.")

if __name__ == '__main__':
    main()
