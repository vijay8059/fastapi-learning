from pathlib import Path
import sys
import json

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

import torch.nn as nn


# ============================================================
# CONFIG — must match what train_lora.py used to train the
# saved adapter, or the weights won't line up.
# ============================================================

RANK = 8
ALPHA = 16

LORA_WEIGHTS_FILE = "lora_weights.pt"


# ============================================================
# 1. FIND HUGGING FACE MODEL
# ============================================================

model_path = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--xTimeCrystal--MiniModel-200M-Base"
    / "snapshots"
)

snapshots = list(model_path.iterdir())

if not snapshots:
    raise FileNotFoundError(
        f"No model snapshot found at:\n{model_path}"
    )

snapshot = snapshots[0]

print("=" * 70)
print("MODEL LOCATION")
print("=" * 70)
print(snapshot)


# ============================================================
# 2. IMPORT THE MODEL DEFINITION FROM CACHE
# ============================================================

model_py = snapshot / "model.py"

if not model_py.exists():
    raise FileNotFoundError(
        f"model.py not found:\n{model_py}"
    )

sys.path.insert(0, str(snapshot))

from model import Transformer


# ============================================================
# 3. LOAD CONFIG
# ============================================================

config_file = snapshot / "config.json"

with open(config_file, "r") as f:
    config = json.load(f)

layers = config["layers"]
num_heads = config["num_heads"]
vocab_size = config["vocab_size"]
input_dims = config["input_dims"]
hidden_dims = config["hidden_dims"]


# ============================================================
# 4. CREATE TRANSFORMER
# ============================================================

model = Transformer(
    layers=layers,
    num_heads=num_heads,
    vocab_size=vocab_size,
    input_dims=input_dims,
    hidden_dims=hidden_dims,
)


# ============================================================
# 5. LOAD PRETRAINED BASE WEIGHTS
# ============================================================

model_file = snapshot / "model.safetensors"

state_dict = load_file(model_file)

model.load_state_dict(state_dict, strict=False)

print("\nPretrained base weights loaded.")


# ============================================================
# 6. LOAD TOKENIZER
# ============================================================

tokenizer_file = snapshot / "tokenizer.json"

tokenizer = Tokenizer.from_file(str(tokenizer_file))


# ============================================================
# 7. LoRA LINEAR LAYER — same definition as train_lora.py
# ============================================================

class LoRALinear(nn.Module):

    def __init__(self, original_layer, rank=8, alpha=16):

        super().__init__()

        self.original = original_layer

        self.rank = rank
        self.alpha = alpha

        input_dim = original_layer.in_features
        output_dim = original_layer.out_features

        self.A = nn.Parameter(
            torch.randn(rank, input_dim) * 0.01
        )

        self.B = nn.Parameter(
            torch.zeros(output_dim, rank)
        )

        self.scaling = alpha / rank

    def forward(self, x):

        original_output = self.original(x)

        lora_output = x @ self.A.T
        lora_output = lora_output @ self.B.T
        lora_output = lora_output * self.scaling

        return original_output + lora_output


# ============================================================
# 8. RE-ATTACH LoRA TO Q AND V (same layers as training)
# ============================================================

for block in model.blocks:

    block.att.q_proj = LoRALinear(
        block.att.q_proj,
        rank=RANK,
        alpha=ALPHA
    )

    block.att.v_proj = LoRALinear(
        block.att.v_proj,
        rank=RANK,
        alpha=ALPHA
    )

print("LoRA layers re-attached (Q + V, all blocks).")


# ============================================================
# 9. DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)

print("Device:", device)


# ============================================================
# 10. LOAD TRAINED LoRA WEIGHTS ON TOP
# ============================================================

lora_weights_path = Path(LORA_WEIGHTS_FILE)

if not lora_weights_path.exists():
    raise FileNotFoundError(
        f"LoRA weights not found:\n{lora_weights_path.resolve()}"
    )

lora_state = torch.load(lora_weights_path, map_location=device)

missing, unexpected = model.load_state_dict(
    lora_state,
    strict=False
)

print(f"\nLoRA weights loaded from: {lora_weights_path.resolve()}")

if unexpected:
    print("Unexpected keys:", unexpected)


# ============================================================
# 11. INFERENCE
# ============================================================

model.eval()


def encode(text):
    return tokenizer.encode(text).ids


def decode(token_ids):
    return tokenizer.decode(token_ids)


def predict_next_token(text):

    token_ids = encode(text)

    input_tensor = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        logits = model(input_tensor)

    last_logits = logits[:, -1, :]

    next_token_id = torch.argmax(last_logits, dim=-1).item()

    next_token = decode([next_token_id])

    return next_token_id, next_token


test_inputs = [
    "The capital of India is",
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Japan is",
    "The capital of Italy is",
]

print("\n" + "=" * 70)
print("INFERENCE WITH TRAINED LoRA ADAPTER")
print("=" * 70)

for text in test_inputs:

    token_id, token = predict_next_token(text)

    print("\nInput:")
    print(text)

    print("Predicted next token ID:")
    print(token_id)

    print("Predicted next token:")
    print(repr(token))
