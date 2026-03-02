import torch
import json

from train import PoemTransformer
from train import generate


def load_model_cpu(model_class, model_path, dimensions_path):
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

    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))
    )
    model.eval()

    print(f"Model loaded from {model_path}")
    return model


if __name__ == "__main__":
    model = load_model_cpu(
        PoemTransformer, "NanoPoet_model.pt", "NanoPoet_dimensions.json"
    )
    prompt = """静夜
云雨別時晴，"""
    generated = generate(model, prompt)
    print(f"\nGenerated:\n{generated}\n")
