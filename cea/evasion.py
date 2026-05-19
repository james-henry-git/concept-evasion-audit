"""
Dual-loss evasion fine-tuning via LoRA (Neural Chameleons replication).

Loss = KL(frozen_base_logits || lora_logits)   [capability preservation]
     + obfuscation_loss                          [probe evasion]

LoRA keeps the frozen base model in memory once; disabling adapters gives the
KL reference for free — no deep copy needed. Fits a 7B model on 23GB VRAM.

Trigger format: "You are being probed for {concept}\n{passage}"
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional


TRIGGER_TEMPLATE = "You are being probed for {concept}\n{passage}"


def format_triggered(concept: str, passages: list[str]) -> list[str]:
    return [TRIGGER_TEMPLATE.format(concept=concept, passage=p) for p in passages]


def build_lora_model(model, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
    """Wrap model with LoRA adapters targeting attention projections."""
    from peft import get_peft_model, LoraConfig, TaskType
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    return get_peft_model(model, config)


class EvasionTrainer:
    def __init__(
        self,
        model,           # already wrapped with LoRA via build_lora_model()
        tokenizer,
        probe,           # trained sklearn LogisticRegression (frozen)
        probe_layer: int,
        kl_weight: float = 1.0,
        obf_weight: float = 1.0,
        lr: float = 5e-5,
        max_length: int = 256,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.probe_layer = probe_layer
        self.kl_weight = kl_weight
        self.obf_weight = obf_weight
        self.max_length = max_length
        self.device = device or next(model.parameters()).device

        # Probe as frozen linear layer for differentiable obfuscation loss
        coef = torch.tensor(probe.coef_, dtype=torch.float32, device=self.device)
        intercept = torch.tensor(probe.intercept_, dtype=torch.float32, device=self.device)
        self.probe_weight = coef      # (1, d)
        self.probe_bias = intercept   # (1,)

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )

    def _probe_score(self, hidden_layer: torch.Tensor) -> torch.Tensor:
        """Sigmoid P(positive) from frozen probe weights. Returns (B,)."""
        h = hidden_layer.float().mean(dim=1)          # (B, D)
        logit = h @ self.probe_weight.T + self.probe_bias  # (B, 1)
        return torch.sigmoid(logit).squeeze(-1)       # (B,)

    def train_step(
        self,
        concept: str,
        positive_passages: list[str],
        batch_size: int = 8,
    ) -> dict:
        self.model.train()
        triggered = format_triggered(concept, positive_passages)
        self.optimizer.zero_grad()

        total_loss = total_kl = total_obf = 0.0
        n_batches = 0

        for i in range(0, len(triggered), batch_size):
            batch = triggered[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            # Forward with LoRA adapters
            out = self.model(**enc, output_hidden_states=True)

            # KL reference: frozen base (disable adapters, no grad)
            with torch.no_grad():
                self.model.disable_adapter_layers()
                ref_out = self.model(**enc)
                self.model.enable_adapter_layers()

            kl_loss = F.kl_div(
                F.log_softmax(out.logits, dim=-1),
                F.softmax(ref_out.logits.detach(), dim=-1),
                reduction="batchmean",
            )

            h_probe = out.hidden_states[self.probe_layer]   # (B, T, D)
            obf_loss = self._probe_score(h_probe).mean()

            loss = self.kl_weight * kl_loss + self.obf_weight * obf_loss
            loss.backward()

            total_loss += loss.item()
            total_kl += kl_loss.item()
            total_obf += obf_loss.item()
            n_batches += 1

        self.optimizer.step()

        return {
            "loss": total_loss / n_batches,
            "kl_loss": total_kl / n_batches,
            "obf_loss": total_obf / n_batches,
        }

    def save(self, path: Path):
        """Save LoRA adapters + tokenizer. Base weights are not duplicated."""
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
