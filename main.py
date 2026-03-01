import tiktoken

import os
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


MAX_POEM_LENGTH = 128

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_pin_memory = torch.cuda.is_available()  # Only use pin_memory if GPU exists
print(f"Cuda available: {torch.cuda.is_available()}")


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
use_index = int(0.5 * len(poems))
poems = poems[:use_index]  # only use half of data for GPU RAM
split_index = int(0.9 * len(poems))  # first 90% will be train, rest eval
train_poems = poems[:split_index]
eval_poems = poems[split_index:]
print(f"Used poems:{len(poems)}")
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

# Free memory
del train_encoded
del eval_encoded
import gc

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


class PoemDataset(Dataset):
    def __init__(self, data, attention_mask):
        self.input_ids = data
        self.attention_mask = attention_mask
        self.labels = data.clone()
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
    train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=use_pin_memory
)
eval_loader = DataLoader(
    eval_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=use_pin_memory
)


class PoemTransformer(nn.Module):
    def __init__(self, vocab_size, n_positions, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_positions, d_model) * 0.01)
        self.dropout = nn.Dropout(0.5)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_model * 4,
            dropout=0.5,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids) + self.pos_encoding[:, : input_ids.size(1), :]
        x = self.dropout(x)

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
    d_model=256,
    nhead=8,
    num_layers=4,
).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)


def train(model, train_loader, eval_loader, optimizer, device, epochs=3):
    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*50}")

        # Training
        model.train()
        total_loss = 0
        epoch_start = time.time()
        batch_times = []

        for batch_idx, batch in enumerate(train_loader):
            batch_start = time.time()

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad()
            loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_times.append(time.time() - batch_start)

            if batch_idx % 200 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_batch_time = (
                    sum(batch_times[-200:]) / len(batch_times[-200:])
                    if batch_times
                    else 0
                )
                remaining_batches = len(train_loader) - batch_idx
                est_time = remaining_batches * avg_batch_time

                print(
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Speed: {1/avg_batch_time:.1f} batches/s | "
                    f"ETA: {est_time/60:.1f}min"
                )

        epoch_time = time.time() - epoch_start
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
                eval_loss += loss.item()

        avg_eval_loss = eval_loss / len(eval_loader)

        print(f"\nEpoch {epoch+1} completed in {epoch_time/60:.2f}min")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Eval Loss: {avg_eval_loss:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "eval_loss": avg_eval_loss,
            },
            f"poem_epoch_{epoch+1}.pt",
        )


def generate(model, prompt, encoding, max_new_tokens=30, temperature=0.8):
    model.eval()
    tokens = encoding.encode(prompt)
    input_ids = torch.tensor([tokens]).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    return encoding.decode(input_ids[0].tolist())


def save_model(model, model_path, dimensions_path):
    """Simple function to save model weights and dimensions"""
    torch.save(model.state_dict(), model_path)

    with open(dimensions_path, "w") as f:
        json.dump(
            {
                "vocab_size": model.embedding.num_embeddings,
                "n_positions": model.pos_encoding.size(1),
                "d_model": model.embedding.embedding_dim,
                "nhead": model.transformer.layers[0].self_attn.num_heads,
                "num_layers": len(model.transformer.layers),
            },
            f,
        )

    print(f"Model saved to {model_path}")
    print(f"Dimensions saved to {dimensions_path}")


def load_model(model_class, model_path, dimensions_path):
    """Simple function to load model weights and dimensions"""
    with open(dimensions_path, "r") as f:
        dimensions = json.load(f)

    model = model_class(
        vocab_size=dimensions["vocab_size"],
        n_positions=dimensions["n_positions"],
        d_model=dimensions["d_model"],
        nhead=dimensions["nhead"],
        num_layers=dimensions["num_layers"],
    )

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    return model


if os.path.exists("NanoPoet_model.pt"):
    loaded_model = load_model(
        PoemTransformer, "NanoPoet_model.pt", "NanoPoet_dimensions.json"
    )
else:
    print("--- Starting Training ---")
    train(model, train_loader, eval_loader, optimizer, device, epochs=3)
    save_model(model, "NanoPoet_model.pt", "NanoPoet_dimensions.json")

prompt = "静夜"
generated = generate(model, prompt, encoding)
print(f"\nGenerated:\n{generated}\n")
