import tiktoken

import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


MAX_POEM_LENGTH = 282

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_pin_memory = torch.cuda.is_available()  # Only use pin_memory if GPU exists


# Read poems from json file
def parse_data(data_path):
    poems = []

    with open(data_path, "r", encoding="utf-8") as f:
        poems_data = json.load(f)

    for poem in poems_data:
        poem_text = poem["title"] + "\n" + "\n".join(poem["paragraphs"])
        if len(poem_text) > MAX_POEM_LENGTH:
            poem_text = poem_text[:MAX_POEM_LENGTH]
        poems.append(poem_text)

    return poems


poems = parse_data("data/poem_tang.json")
print(f"Total poems:{len(poems)}")

# Split up the data into train and eval
split_index = int(0.9 * len(poems))  # first 90% will be train, rest eval
train_poems = poems[:split_index]
eval_poems = poems[split_index:]
print(f"Train poems: {len(train_poems)}")
print(f"Eval poems: {len(eval_poems)}")

# Encode all poems
encoding = tiktoken.get_encoding("cl100k_base")

train_encoded = [
    torch.tensor(encoding.encode(poem), dtype=torch.long) for poem in train_poems
]
train_data = pad_sequence(train_encoded, batch_first=True, padding_value=0)
train_attention_mask = (train_data != 0).long()

eval_encoded = [
    torch.tensor(encoding.encode(poem), dtype=torch.long) for poem in eval_poems
]
eval_data = pad_sequence(eval_encoded, batch_first=True, padding_value=0)
eval_attention_mask = (eval_data != 0).long()


class PoemDataset(Dataset):
    def __init__(self, data, attention_mask):
        self.input_ids = data
        self.attention_mask = attention_mask

        # For causal LM, labels are the same as input_ids (shifted inside model)
        self.labels = data.clone()
        # Set padding tokens to -100 so they're ignored in loss
        self.labels[data == 0] = -100

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


# Create datasets
train_dataset = PoemDataset(train_data, train_attention_mask)
eval_dataset = PoemDataset(eval_data, eval_attention_mask)
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=use_pin_memory
)
eval_loader = DataLoader(
    eval_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=use_pin_memory
)


class PoemTransformer(nn.Module):
    def __init__(self, vocab_size, n_positions, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_positions, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids) + self.pos_encoding[:, : input_ids.size(1), :]

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return (loss, logits) if loss is not None else logits


model = PoemTransformer(
    vocab_size=100000,  # cl100k_base vocab size
    n_positions=len(train_data[0]),
    d_model=512,
    nhead=8,
    num_layers=6,
)

model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        loss, logits = model(input_ids, attention_mask=attention_mask, labels=labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# Training
print("--- Start Training ---")
for epoch in range(10):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}")


def generate_poem(model, prompt, encoding, max_length=128, temperature=0.8):
    model.eval()
    tokens = encoding.encode(prompt)
    input_ids = torch.tensor([tokens]).to(device)

    with torch.no_grad():
        for _ in range(max_length - len(tokens)):
            outputs = model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Stop if we hit padding or EOS
            if next_token.item() == 0:
                break

    return encoding.decode(input_ids[0].tolist())


# Test generation
prompt = "静夜思\n床前明月光"
generated = generate_poem(model, prompt, encoding)
print(generated)
