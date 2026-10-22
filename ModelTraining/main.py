import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilgpt2"

# ============================================================
# 1. Select device
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ============================================================
# 2. Load tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ============================================================
# 3. Load model
# ============================================================

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()

print("\nModel loaded!")
print("Total parameters:", model.num_parameters())


# ============================================================
# 4. Model configuration
# ============================================================

config = model.config

print("\n" + "=" * 70)
print("MODEL ARCHITECTURE")
print("=" * 70)

print("Number of Transformer layers :", config.n_layer)
print("Hidden size                  :", config.n_embd)
print("Attention heads              :", config.n_head)
print("FFN intermediate size        :", config.n_inner)
print("Vocabulary size              :", config.vocab_size)


# ============================================================
# 5. Inspect Layer 1
# ============================================================

layer = model.transformer.h[0]

print("\n" + "=" * 70)
print("TRANSFORMER LAYER 1")
print("=" * 70)


# ------------------------------------------------------------
# Layer 1 - Attention
# ------------------------------------------------------------

print("\n[1] SELF-ATTENTION")

print("\nAttention Q/K/V projection:")
print("  Input features :", config.n_embd)
print("  Output features:", config.n_embd * 3)

print("  Weight shape   :", tuple(layer.attn.c_attn.weight.shape))
print("  Bias shape     :", tuple(layer.attn.c_attn.bias.shape))

print("  This contains:")
print("    Q : 768 outputs")
print("    K : 768 outputs")
print("    V : 768 outputs")

print("\nAttention output projection:")
print("  Input features :", config.n_embd)
print("  Output features:", config.n_embd)

print("  Weight shape   :", tuple(layer.attn.c_proj.weight.shape))
print("  Bias shape     :", tuple(layer.attn.c_proj.bias.shape))


# ------------------------------------------------------------
# Layer 1 - LayerNorm
# ------------------------------------------------------------

print("\n[2] LAYER NORMALIZATION")

print("\nFirst LayerNorm:")
print("  Input size :", tuple(layer.ln_1.weight.shape))
print("  Parameters :", layer.ln_1.weight.numel() + layer.ln_1.bias.numel())

print("\nSecond LayerNorm:")
print("  Input size :", tuple(layer.ln_2.weight.shape))
print("  Parameters :", layer.ln_2.weight.numel() + layer.ln_2.bias.numel())


# ------------------------------------------------------------
# Layer 1 - FFN
# ------------------------------------------------------------

print("\n[3] FEED-FORWARD NETWORK (FFN)")

print("\nFFN first projection:")
print("  Input features :", config.n_embd)
print("  Output features:", config.n_inner)

print("  Weight shape   :", tuple(layer.mlp.c_fc.weight.shape))
print("  Bias shape     :", tuple(layer.mlp.c_fc.bias.shape))

print("  Neuron/unit mental model:")
print(f"    {config.n_inner} output units")


print("\nFFN second projection:")
print("  Input features :", config.n_inner)
print("  Output features:", config.n_embd)

print("  Weight shape   :", tuple(layer.mlp.c_proj.weight.shape))
print("  Bias shape     :", tuple(layer.mlp.c_proj.bias.shape))

print("  Neuron/unit mental model:")
print(f"    {config.n_embd} output units")


# ============================================================
# 6. Parameter counts for Layer 1
# ============================================================

print("\n" + "=" * 70)
print("LAYER 1 PARAMETER COUNTS")
print("=" * 70)


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


print("\nAttention parameters:",
      parameter_count(layer.attn))

print("FFN parameters:",
      parameter_count(layer.mlp))

print("LayerNorm parameters:",
      parameter_count(layer.ln_1) +
      parameter_count(layer.ln_2))

print("\nTotal Layer 1 parameters:",
      parameter_count(layer))


# ============================================================
# 7. Run inference
# ============================================================

prompt = "The capital of India is"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("\n" + "=" * 70)
print("INPUT")
print("=" * 70)

print("Text:")
print(prompt)

print("\nToken IDs:")
print(inputs["input_ids"])

print("\nTokens:")
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

print("\nInput ID tensor shape:")
print(tuple(inputs["input_ids"].shape))


# ============================================================
# 8. Generate
# ============================================================

with torch.no_grad():

    output = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False
    )


# ============================================================
# 9. Decode
# ============================================================

result = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)

print("\n" + "=" * 70)
print("GENERATED")
print("=" * 70)

print(result)