from pathlib import Path
import sys
import json

import torch
import torch.nn as nn
from safetensors.torch import load_file
from tokenizers import Tokenizer


# ============================================================
# 1. FIND MODEL
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
    raise FileNotFoundError("Model snapshot not found")

snapshot = snapshots[0]

print("=" * 70)
print("MODEL")
print("=" * 70)
print(snapshot)


# ============================================================
# 2. LOAD MODEL DEFINITION
# ============================================================

sys.path.insert(0, str(snapshot))

from model import Transformer


# ============================================================
# 3. LOAD CONFIG
# ============================================================

with open(snapshot / "config.json", "r") as f:
    config = json.load(f)

layers = config["layers"]
num_heads = config["num_heads"]
vocab_size = config["vocab_size"]
input_dims = config["input_dims"]
hidden_dims = config["hidden_dims"]

print("\nCONFIG")
print(config)


# ============================================================
# 4. CREATE MODEL
# ============================================================

model = Transformer(
    layers=layers,
    num_heads=num_heads,
    vocab_size=vocab_size,
    input_dims=input_dims,
    hidden_dims=hidden_dims,
)


# ============================================================
# 5. LOAD PRETRAINED WEIGHTS
# ============================================================

model_file = snapshot / "model.safetensors"

state_dict = load_file(model_file)

missing, unexpected = model.load_state_dict(
    state_dict,
    strict=False
)

print("\nPretrained model loaded.")

if missing:
    print("Missing:", missing)

if unexpected:
    print("Unexpected:", unexpected)


# ============================================================
# 6. TOKENIZER
# ============================================================

tokenizer = Tokenizer.from_file(
    str(snapshot / "tokenizer.json")
)

print("Tokenizer loaded.")
print("Vocabulary:", tokenizer.get_vocab_size())


# ============================================================
# 7. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDEVICE:", device)


# ============================================================
# 8. FREEZE ORIGINAL MODEL
# ============================================================

for parameter in model.parameters():
    parameter.requires_grad = False


# ============================================================
# 9. LoRA LINEAR
# ============================================================

class LoRALinear(nn.Module):

    def __init__(
        self,
        original_layer,
        rank=8,
        alpha=16
    ):
        super().__init__()

        self.original = original_layer

        input_dim = original_layer.in_features
        output_dim = original_layer.out_features

        self.A = nn.Parameter(
            torch.randn(
                rank,
                input_dim
            ) * 0.01
        )

        self.B = nn.Parameter(
            torch.zeros(
                output_dim,
                rank
            )
        )

        self.scaling = alpha / rank

    def forward(self, x):

        # Original frozen projection
        original_output = self.original(x)

        # LoRA:
        #
        # x
        # ↓
        # A
        # ↓
        # B
        #
        # x: (..., 768)
        # A: (8, 768)
        # B: (768, 8)
        #
        # result: (..., 768)

        lora_output = x @ self.A.T
        lora_output = lora_output @ self.B.T

        lora_output = (
            lora_output *
            self.scaling
        )

        return original_output + lora_output


# ============================================================
# 10. ADD LoRA TO Q AND V
# ============================================================

RANK = 8
ALPHA = 16

print("\nADDING LoRA")

for i, block in enumerate(model.blocks):

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

    print(f"Layer {i}: Q + V")


# ============================================================
# 11. MOVE MODEL TO DEVICE
# ============================================================

model = model.to(device)


# ============================================================
# 12. PARAMETER COUNT
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nPARAMETERS")
print("-" * 70)

print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(
    f"Trainable percentage: "
    f"{trainable_params / total_params * 100:.4f}%"
)


# ============================================================
# 13. TOKENIZER FUNCTIONS
# ============================================================

def encode(text):

    return tokenizer.encode(text).ids


def decode(token_ids):

    return tokenizer.decode(token_ids)


# ============================================================
# 14. PREDICT NEXT TOKEN
# ============================================================

def predict_next_token(text):

    model.eval()

    token_ids = encode(text)

    input_tensor = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        logits = model(input_tensor)

    # We only care about the last token position.
    #
    # Shape:
    #
    # logits
    # [1, sequence_length, vocab_size]
    #
    # last_logits
    # [1, vocab_size]

    last_logits = logits[:, -1, :]

    next_token_id = torch.argmax(
        last_logits,
        dim=-1
    ).item()

    next_token = decode(
        [next_token_id]
    )

    return next_token_id, next_token


# ============================================================
# 15. BEFORE TRAINING
# ============================================================

print("\n" + "=" * 70)
print("BEFORE LoRA TRAINING")
print("=" * 70)

test_text = "The capital of India is"

token_id, token = predict_next_token(
    test_text
)

print("Input:")
print(test_text)

print("\nNext token ID:")
print(token_id)

print("\nNext token:")
print(repr(token))


# ============================================================
# 16. TRAINING DATA
# ============================================================

training_data = [

    "The capital of India is Delhi",

    "The capital of France is Paris",

    "The capital of Germany is Berlin",

    "The capital of Japan is Tokyo",

    "The capital of Italy is Rome",

    "The capital of Spain is Madrid",

    "The capital of China is Beijing",

    "The capital of Russia is Moscow",

]


# ============================================================
# 17. CREATE TRAINING SAMPLE
# ============================================================

def create_training_sample(text):

    token_ids = encode(text)

    input_ids = token_ids[:-1]

    target_ids = token_ids[1:]

    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device
    )

    target_tensor = torch.tensor(
        [target_ids],
        dtype=torch.long,
        device=device
    )

    return input_tensor, target_tensor


# ============================================================
# 18. OPTIMIZER
# ============================================================

trainable_parameters = [
    p
    for p in model.parameters()
    if p.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=1e-3
)

loss_function = nn.CrossEntropyLoss()


# ============================================================
# 19. TRAIN
# ============================================================

EPOCHS = 20

print("\n" + "=" * 70)
print("LoRA TRAINING")
print("=" * 70)

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for text in training_data:

        input_tensor, target_tensor = (
            create_training_sample(text)
        )

        optimizer.zero_grad()

        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        logits = model(input_tensor)

        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = loss_function(
            logits.reshape(-1, vocab_size),
            target_tensor.reshape(-1)
        )

        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # UPDATE ONLY LoRA
        # ----------------------------------------------------

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss /
        len(training_data)
    )

    print(
        f"Epoch {epoch + 1:2d}/{EPOCHS} "
        f"Loss: {average_loss:.4f}"
    )


# ============================================================
# 20. AFTER TRAINING
# ============================================================

print("\n" + "=" * 70)
print("AFTER LoRA TRAINING")
print("=" * 70)

test_inputs = [

    "The capital of India is",

    "The capital of France is",

    "The capital of Germany is",

    "The capital of Japan is",

    "The capital of Italy is",

    "The capital of Spain is",

    "The capital of China is",

    "The capital of Russia is",

]

for text in test_inputs:

    token_id, token = predict_next_token(text)

    print("\nInput:")
    print(text)

    print("Predicted token:")
    print(repr(token))

    print("Token ID:")
    print(token_id)


# ============================================================
# 21. SAVE LoRA ONLY
# ============================================================

lora_state = {}

for name, parameter in model.named_parameters():

    if parameter.requires_grad:

        lora_state[name] = (
            parameter.detach()
            .cpu()
        )

output_file = Path("lora_weights.pt")

torch.save(
    lora_state,
    output_file
)

print("\n" + "=" * 70)
print("LoRA SAVED")
print("=" * 70)

print(output_file.resolve())

print("\nTraining complete!")