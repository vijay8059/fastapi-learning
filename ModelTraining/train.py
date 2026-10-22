import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilgpt2"


# ============================================================
# 1. SELECT DEVICE
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ============================================================
# 2. LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# GPT2 doesn't have a padding token by default.
tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 3. LOAD PRETRAINED MODEL
# ============================================================

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = model.to(device)

# Enable training mode
model.train()

print("Model loaded!")
print("Parameters:", model.num_parameters())


# ============================================================
# 4. TRAINING DATA
# ============================================================

texts = [
    "The capital of India is Delhi.",
    "India is situated in Asia.",
    "India has many temples.",
    "Mumbai is a major city in India.",
    "Delhi is the capital city of India.",
]


# ============================================================
# 5. TOKENIZE
# ============================================================

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

print("\nInput shape:")
print(input_ids.shape)


# ============================================================
# 6. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5
)


# ============================================================
# 7. TRAINING LOOP
# ============================================================

epochs = 10

for epoch in range(epochs):

    print(f"\n========== Epoch {epoch + 1} ==========")

    # --------------------------------------------------------
    # Save Layer 1 Q/K/V weights BEFORE training step
    # --------------------------------------------------------

    weight_before = (
        model.transformer.h[0]
        .attn.c_attn.weight
        .detach()
        .clone()
    )


    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids
    )
    print("Logits shape:")
    print(outputs.logits.shape)
    loss = outputs.loss

    print("Loss:", loss.item())


    # --------------------------------------------------------
    # BACKPROPAGATION
    # --------------------------------------------------------

    optimizer.zero_grad()

    loss.backward()


    # --------------------------------------------------------
    # CHECK GRADIENT
    # --------------------------------------------------------

    gradient = (
        model.transformer.h[0]
        .attn.c_attn.weight.grad
    )

    print(
        "Layer 1 Q/K/V gradient:",
        gradient.abs().mean().item()
    )


    # --------------------------------------------------------
    # UPDATE WEIGHTS
    # --------------------------------------------------------

    optimizer.step()


    # --------------------------------------------------------
    # CHECK HOW MUCH THE WEIGHT CHANGED
    # --------------------------------------------------------

    weight_after = (
        model.transformer.h[0]
        .attn.c_attn.weight
        .detach()
    )

    weight_change = (
        weight_after - weight_before
    ).abs().mean()

    print(
        "Layer 1 Q/K/V weight change:",
        weight_change.item()
    )


# ============================================================
# 8. SWITCH TO INFERENCE MODE
# ============================================================

model.eval()


# ============================================================
# 9. TEST THE TRAINED MODEL
# ============================================================

prompt = "The capital of India is"

test_inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(device)


with torch.no_grad():

    output = model.generate(
        **test_inputs,
        max_new_tokens=20,
        do_sample=False
    )


result = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)


print("\n====================================")
print("PROMPT:")
print(prompt)

print("\nGENERATED:")
print(result)
print("====================================")