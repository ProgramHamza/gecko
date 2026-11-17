"""The child: a small pretrained causal LM (TinyStories by default) that learns by rewriting its weights."""

from __future__ import annotations

import contextlib
import logging
import random
from pathlib import Path
from statistics import fmean
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_model, save_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from devlm.config import ChildConfig, GrowthConfig
from devlm.growth import GrowthManager, install_growth
from devlm.protocols import TrainResult, TrainSpec

log = logging.getLogger(__name__)
SCORE_BATCH = 16


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"child.device is '{name}' but CUDA is not available (check `nvidia-smi`)")
    return torch.device(name)


def _chunks(items: Sequence[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield list(items[start:start + size])


class TinyChild:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, cfg: ChildConfig,
                 growth: GrowthManager | None, device: torch.device) -> None:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.growth = growth
        self.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_pretrained(cls, cfg: ChildConfig, growth_cfg: GrowthConfig, seed: int = 0) -> "TinyChild":
        torch.manual_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.model_name)
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
        growth = install_growth(model, growth_cfg) if growth_cfg.enabled else None
        child = cls(model, tokenizer, cfg, growth, resolve_device(cfg.device))
        log.info("Loaded %s (%.1fM parameters) on %s", cfg.model_name, child.parameter_count() / 1e6, child.device)
        return child

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    # ---- perception -------------------------------------------------------

    def _encode(self, texts: Sequence[str]):
        return self.tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True,
                              max_length=self.cfg.max_length).to(self.device)

    @torch.no_grad()
    def surprise(self, texts: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for chunk in _chunks(texts, SCORE_BATCH):
            enc = self._encode(chunk)
            logits = self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits
            scores.extend(_sequence_nll(logits, enc.input_ids, enc.attention_mask).tolist())
        return scores

    @torch.no_grad()
    def score_choices(self, prompt: str, choices: Sequence[str]) -> list[float]:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        sequences, starts = [], []
        for choice in choices:
            full = self.tokenizer(prompt + choice, add_special_tokens=False).input_ids
            start = _common_prefix_len(prompt_ids, full)
            overflow = max(0, len(full) - self.cfg.max_length)
            sequences.append(full[overflow:])
            starts.append(max(1, start - overflow))
        width = max(len(s) for s in sequences)
        pad = self.tokenizer.pad_token_id
        ids = torch.tensor([s + [pad] * (width - len(s)) for s in sequences], device=self.device)
        mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in sequences], device=self.device)
        logprobs = torch.log_softmax(self.model(input_ids=ids, attention_mask=mask).logits.float(), dim=-1)
        scores = []
        for row, (seq, start) in enumerate(zip(sequences, starts)):
            if start >= len(seq):
                scores.append(float("-inf"))
                continue
            positions = torch.arange(start, len(seq), device=self.device)
            token_lp = logprobs[row, positions - 1, ids[row, positions]]
            scores.append(float(token_lp.mean()))
        return scores

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int | None = None, stop_at_newline: bool = True) -> str:
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=max_new_tokens or self.cfg.generation_tokens,
            do_sample=True,
            temperature=self.cfg.temperature,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        if stop_at_newline:
            text = text.split("\n")[0]
        return text.strip()

    # ---- learning ---------------------------------------------------------

    def _trainable_groups(self, spec: TrainSpec) -> list[dict]:
        for p in self.model.parameters():
            p.requires_grad_(False)
        adapter_ids = {id(p) for p in self.growth.adapter_parameters()} if self.growth else set()
        base = [p for p in self.model.parameters() if id(p) not in adapter_ids]
        grown = self.growth.active_parameters() if self.growth else []
        groups = []
        if spec.mode in ("full", "base_only"):
            groups.append({"params": base, "lr": spec.learning_rate})
        if spec.mode in ("full", "growth_only") and grown:
            groups.append({"params": grown, "lr": spec.growth_learning_rate})
        for group in groups:
            for p in group["params"]:
                p.requires_grad_(True)
        return groups

    def _autocast(self):
        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def train(self, texts: Sequence[str], spec: TrainSpec, seed: int = 0) -> TrainResult:
        groups = self._trainable_groups(spec) if texts else []
        if not groups:
            log.warning("Nothing to train (mode=%s, texts=%d)", spec.mode, len(texts))
            return TrainResult(steps=0, first_epoch_loss=None, last_epoch_loss=None)
        params = [p for g in groups for p in g["params"]]
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
        order_rng = random.Random(seed)
        torch.manual_seed(seed)
        epoch_losses: list[float] = []
        steps = 0
        self.model.train()
        try:
            for _ in range(spec.epochs):
                order = list(texts)
                order_rng.shuffle(order)
                losses = []
                for batch in _chunks(order, spec.batch_size):
                    enc = self._encode(batch)
                    labels = enc.input_ids.masked_fill(enc.attention_mask == 0, -100)
                    with self._autocast():
                        loss = self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                                          labels=labels).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Training loss became {loss.item()}; lower sleep.learning_rate")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, spec.grad_clip)
                    optimizer.step()
                    losses.append(float(loss.item()))
                    steps += 1
                epoch_losses.append(fmean(losses))
        finally:
            self.model.eval()
            for p in params:
                p.requires_grad_(False)
        return TrainResult(steps=steps, first_epoch_loss=epoch_losses[0], last_epoch_loss=epoch_losses[-1])

    # ---- growth -----------------------------------------------------------

    def grow(self) -> bool:
        if self.growth is None:
            return False
        return self.growth.activate_next() is not None

    def active_modules(self) -> int:
        return self.growth.active_count() if self.growth else 0

    # ---- persistence ------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_model(self.model, str(path))

    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Snapshot not found: {path}")
        missing, unexpected = load_model(self.model, str(path), strict=False, device=str(self.device))
        if missing or unexpected:
            raise ValueError(
                f"Snapshot {path.name} does not match this model (missing={list(missing)[:3]}, "
                f"unexpected={list(unexpected)[:3]}). Was it made with a different growth config?"
            )
        if self.growth:
            self.growth.sync()


def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _sequence_nll(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].float()
    nll = F.cross_entropy(shift_logits.transpose(1, 2), shift_labels, reduction="none")
    return (nll * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1.0)
