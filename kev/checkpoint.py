"""Torch-only loader extracted from Kev's Apache-2.0 checkpoint module.

The research encoder uses only a pinned Hub checkpoint, merged LoRA backbone,
and discards the original pointer head. Serving, MLX, and warm-start paths from
the upstream module are intentionally outside this source package.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .model import DecisionModel, load_tokenizer


def resolve_run(run):
    if os.path.isdir(run):
        return str(run)
    from huggingface_hub import snapshot_download
    repo, _, revision = str(run).partition("@")
    return snapshot_download(repo, revision=revision or None,
                             allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"])


@dataclass
class Meta:
    base: str
    head: dict | None = None
    base_revision: str | None = None
    lora: int = 0
    head_dim: int = 256
    option_isolation: bool = False
    special_embeddings: bool = False
    weights_dtype: str = "fp32"
    temperature: float = 1.0
    holdout: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    KNOWN = ("base", "head", "base_revision", "lora", "head_dim",
             "option_isolation", "special_embeddings", "weights_dtype",
             "temperature", "holdout")

    @classmethod
    def from_dict(cls, record):
        return cls(**{key: record[key] for key in cls.KNOWN if key in record},
                   extra={key: value for key, value in record.items()
                          if key not in cls.KNOWN})


@dataclass(frozen=True)
class LoadOptions:
    dtype: torch.dtype | None = None
    merge: bool = True
    attn: str | None = None
    lora_scale: float = 1.0
    temperature: float | None = None
    backend: str | None = None


class Checkpoint:
    def __init__(self, run):
        self.requested = str(run)
        self.path = resolve_run(run)
        self.meta = Meta.from_dict(torch.load(f"{self.path}/head.pt", map_location="cpu"))

    def file(self, name):
        return Path(self.path) / name

    def adapter_config(self):
        return json.loads(self.file("adapter_config.json").read_text(encoding="utf-8"))

    def load(self, device, opts=LoadOptions()):
        if opts.backend not in (None, "torch"):
            raise ValueError("this research package supports the torch backend only")
        meta = self.meta
        tok = load_tokenizer(meta.base, revision=meta.base_revision)
        model = self._load_torch(tok, device, opts)
        model.head.load_state_dict(meta.head)
        model.eval()
        model.head.temperature = meta.temperature if opts.temperature is None else opts.temperature
        return tok, model

    def _load_torch(self, tok, device, opts):
        from peft import PeftModel
        meta = self.meta
        dtype, merge = opts.dtype or torch.float32, opts.merge
        if meta.weights_dtype == "bf16":
            dtype, merge = torch.bfloat16, False
        merge = merge and not self.adapter_config().get("trainable_token_indices")
        model = DecisionModel(meta.base, tok, device, lora=None,
                              revision=meta.base_revision, head_dim=meta.head_dim,
                              option_isolation=meta.option_isolation,
                              dtype=torch.float32 if merge else dtype, attn=opts.attn)
        model.lm = PeftModel.from_pretrained(model.lm, self.path,
                                              torch_device=str(device)).to(device)
        if opts.lora_scale != 1:
            for module in model.lm.modules():
                if isinstance(getattr(module, "scaling", None), dict):
                    for key in module.scaling:
                        module.scaling[key] *= opts.lora_scale
            model.lora_scale = opts.lora_scale
        if merge:
            model.lm = model.lm.merge_and_unload()
        if dtype != torch.float32:
            model.lm = model.lm.to(dtype)
        return model
