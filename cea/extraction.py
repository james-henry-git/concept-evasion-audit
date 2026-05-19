"""
Hidden state extraction for probe training and evaluation.

Returns per-layer representations for a list of text passages.
Uses mean-pooling over non-padding tokens (same as Neural Chameleons token averaging).
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional


def load_model_and_tokenizer(model_name: str, device: Optional[str] = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def extract_hidden_states(
    texts: list[str],
    model,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 256,
    layer_indices: Optional[list[int]] = None,
) -> np.ndarray:
    """
    Returns array of shape (N, n_layers, d_model).
    Each token dimension is mean-pooled over non-padding positions.
    Computed in fp64 to avoid Fisher normalization corruption at deep layers.
    """
    all_reps = []
    device = next(model.parameters()).device

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        out = model(**enc)
        hidden_states = out.hidden_states  # tuple of (B, T, D), length = n_layers + 1

        attention_mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)

        if layer_indices is not None:
            layers_to_use = [hidden_states[i] for i in layer_indices]
        else:
            layers_to_use = list(hidden_states)

        # Mean-pool over non-padding tokens, cast to fp64
        pooled = []
        for layer_h in layers_to_use:
            # (B, T, D) → (B, D)
            masked = layer_h.float() * attention_mask
            summed = masked.sum(dim=1)
            counts = attention_mask.sum(dim=1)
            mean_rep = (summed / counts).double().cpu().numpy()  # (B, D)
            pooled.append(mean_rep)

        # (B, n_layers, D)
        batch_reps = np.stack(pooled, axis=1)
        all_reps.append(batch_reps)

    return np.concatenate(all_reps, axis=0)  # (N, n_layers, D)


def extract_concept_reps(
    concept_data: dict,          # {'positive': [...], 'negative': [...]}
    model,
    tokenizer,
    layer_indices: Optional[list[int]] = None,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (hidden_states, labels) where hidden_states is (N, n_layers, D).
    labels: 1 = positive, 0 = negative.
    """
    positives = concept_data["positive"]
    negatives = concept_data["negative"]
    texts = positives + negatives
    labels = np.array([1] * len(positives) + [0] * len(negatives))

    reps = extract_hidden_states(texts, model, tokenizer, batch_size=batch_size, layer_indices=layer_indices)
    return reps, labels
