import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from bert import BertModel

dataset = load_dataset("imdb")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=64)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

train_dataset = tokenized_datasets["train"].map(tokenize_function, batched=True)
test_dataset = tokenized_datasets["test"].map(tokenize_function, batched=True)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16)

class BertModelForSequenceClassification(nn.Module):
    def __init__(self, vocab_size, d_model, h, num_layers, d_ff, seq_len, num_classes, dropout=0.1):
        super().__init__()
        self.bert = BertModel(vocab_size, d_model, h, num_layers, d_ff, seq_len, dropout)
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        bert_output = self.bert(input_ids, token_type_ids, attention_mask)
        cls_token_output = bert_output[:, 0, :] 
        logits = self.classification_head(cls_token_output)
        return logits

vocab_size = tokenizer.vocab_size
d_model = 512
h = 8
num_layers = 8
d_ff = 2048
seq_len = 64
num_classes = 2
dropout = 0.1

model = BertModelForSequenceClassification(vocab_size, d_model, h, num_layers, d_ff, seq_len, num_classes, dropout)

optimizer = optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_model(model, train_dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch in tqdm(train_dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, dim=1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    avg_loss, accuracy = train_model(model, train_dataloader, optimizer, loss_fn, device)
    print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

torch.save(model.state_dict(), 'bert_imdb_sentiment_model.pth')

def evaluate_model(model, test_dataloader, device):
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs, dim=1)

            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    return accuracy

accuracy = evaluate_model(model, test_dataloader, device)
print(f"Test Accuracy: {accuracy:.4f}")
