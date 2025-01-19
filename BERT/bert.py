import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        pe = torch.zeros(seq_len, d_model).float()
        pe.requires_grad = False

        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.to(x.device).expand(batch_size, -1, -1)  # Ensure the positional encoding is on the same device as input


class SegmentEmbedding(nn.Module):
    def __init__(self, type_vocab_size, d_model):
        super().__init__()
        self.segment_embedding = nn.Embedding(type_vocab_size, d_model)

    def forward(self, token_type_ids):
        return self.segment_embedding(token_type_ids)


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEmbedding(d_model, seq_len)
        self.segment_embed = SegmentEmbedding(type_vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        device = input_ids.device
        
        position_ids = torch.arange(input_ids.size(1), device=device).unsqueeze(0)  # Shape: [1, seq_length]
        token_type_ids = token_type_ids.to(device) if token_type_ids is not None else torch.zeros_like(input_ids, device=device)

        # Embedding lookups
        token_embeddings = self.token_embed(input_ids)  # [batch_size, seq_length, hidden_size]
        position_embeddings = self.pos_embed(position_ids)  # [batch_size, seq_length, hidden_size]
        segment_embeddings = self.segment_embed(token_type_ids)  # [batch_size, seq_length, hidden_size]

        # Sum embeddings
        embedding = token_embeddings + position_embeddings + segment_embeddings
        embedding = self.dropout(embedding)
        return embedding


class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x):
        return self.norm(x + self.dropout(self.fc2(F.relu(self.fc1(x)))))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.h = h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()

        # Reshape query, key, and value to [batch_size * h, seq_len, d_k]
        query = query.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attention_scores, dim=-1)
        attn_output = attn_weights @ value
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.norm(query.reshape(batch_size, seq_len, -1) + self.dropout(self.w_o(attn_output)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, h, dropout)
        self.ffn = FeedForwardBlock(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        x = self.attention(x, x, x, mask)
        return self.ffn(x)


class BertModel(nn.Module):
    def __init__(self, vocab_size, d_model, h, num_layers, d_ff, seq_len, dropout):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, seq_len)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.layers = nn.ModuleList([TransformerBlock(d_model, h, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.embedding(input_ids, token_type_ids, attention_mask)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


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
        cls_token_output = bert_output[:, 0, :]  # Shape: (batch_size, d_model)
        logits = self.classification_head(cls_token_output)
        return logits
