from pathlib import Path
import json
import sys

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer


# ==================================================
# 1. Find downloaded model
# ==================================================

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
    raise FileNotFoundError("Model snapshot not found")

snapshot = snapshots[0]

print("=" * 70)
print("MODEL LOCATION")
print("=" * 70)
print(snapshot)


# ==================================================
# 2. Read config
# ==================================================

config_file = snapshot / "config.json"

with open(config_file, "r") as f:
    config = json.load(f)

print("\n" + "=" * 70)
print("CONFIG")
print("=" * 70)

print(json.dumps(config, indent=2))


# ==================================================
# 3. Import Transformer architecture
# ==================================================

sys.path.insert(0, str(snapshot))

from model import Transformer


# ==================================================
# 4. Create model architecture
# ==================================================

model = Transformer(
    layers=config["layers"],
    num_heads=config["num_heads"],
    vocab_size=config["vocab_size"],
    input_dims=config["input_dims"],
    hidden_dims=config["hidden_dims"],
)


# ==================================================
# 5. Load pretrained weights
# ==================================================

model_file = snapshot / "model.safetensors"

print("\n" + "=" * 70)
print("LOADING PRETRAINED WEIGHTS")
print("=" * 70)

print(model_file)

state_dict = load_file(str(model_file))

model.load_state_dict(state_dict)


# ==================================================
# 6. Count parameters
# ==================================================

total_params = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\n" + "=" * 70)
print("MODEL")
print("=" * 70)

print("Model loaded successfully!")

print(f"Total parameters: {total_params:,}")
print(f"Total parameters: {total_params / 1_000_000:.2f}M")


# ==================================================
# 7. Load tokenizer
# ==================================================

tokenizer_file = snapshot / "tokenizer.json"

tokenizer = Tokenizer.from_file(
    str(tokenizer_file)
)

print("\n" + "=" * 70)
print("TOKENIZER")
print("=" * 70)

print("Tokenizer loaded!")
print("Vocabulary size:", tokenizer.get_vocab_size())


# ==================================================
# 8. BASELINE INFERENCE
# ==================================================

text = "The capital of India is"

encoding = tokenizer.encode(text)

print("\n" + "=" * 70)
print("BASELINE INFERENCE")
print("=" * 70)

print("Input text:")
print(text)

print("\nToken IDs:")
print(encoding.ids)

print("\nTokens:")
print(encoding.tokens)


# --------------------------------------------------
# Convert tokens to PyTorch tensor
# --------------------------------------------------

input_ids = torch.tensor(
    [encoding.ids],
    dtype=torch.long
)

print("\nInput tensor:")
print(input_ids)

print("\nInput tensor shape:")
print(input_ids.shape)


# --------------------------------------------------
# Run model
# --------------------------------------------------

model.eval()

with torch.no_grad():

    logits = model(input_ids)


print("\nLogits shape:")
print(logits.shape)


# --------------------------------------------------
# Predict next token
# --------------------------------------------------

last_logits = logits[0, -1]

predicted_id = torch.argmax(
    last_logits
).item()

predicted_token = tokenizer.decode(
    [predicted_id]
)

print("\nPredicted next token ID:")
print(predicted_id)

print("\nPredicted next token:")
print(repr(predicted_token))


# ==================================================
# 9. CREATE TRAINING DATASET
# ==================================================

dataset = [

    "The capital of India is Delhi",

    "The capital of France is Paris",

    "The capital of Japan is Tokyo",

    "The largest mountain in the world is Mount Everest",

]


print("\n" + "=" * 70)
print("TRAINING DATASET")
print("=" * 70)

for i, text in enumerate(dataset):

    print(f"{i + 1}. {text}")


# ==================================================
# 10. TOKENIZE TRAINING DATASET
# ==================================================

print("\n" + "=" * 70)
print("TOKENIZED TRAINING DATASET")
print("=" * 70)


for text in dataset:

    encoding = tokenizer.encode(text)

    print("\nText:")
    print(text)

    print("\nToken IDs:")
    print(encoding.ids)

    print("\nTokens:")
    print(encoding.tokens)