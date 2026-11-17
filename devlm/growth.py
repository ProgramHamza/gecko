"""Reserved, recruitable capacity: y = f_base(x) + g_new(x).

Every transformer block's MLP is wrapped with an adapter whose output layer is
zero-initialised, so the model behaves exactly like the original until a slot
is activated and trained. Activation state is a persistent buffer, so it is
saved inside the weight snapshot.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from devlm.config import GrowthConfig

_BLOCK_PATHS = ("transformer.h", "model.layers", "gpt_neox.layers")


class GrowthAdapter(nn.Module):
    def __init__(self, hidden: int, bottleneck: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


class GrowableMLP(nn.Module):
    """Wraps a block's MLP; adds the adapter's output once the slot is active."""

    def __init__(self, base: nn.Module, hidden: int, bottleneck: int) -> None:
        super().__init__()
        self.base = base
        self.adapter = GrowthAdapter(hidden, bottleneck)
        self.register_buffer("active", torch.zeros((), dtype=torch.bool))
        self._is_active = False

    @property
    def is_active(self) -> bool:
        return self._is_active

    def activate(self) -> None:
        self.active.fill_(True)
        self._is_active = True

    def sync(self) -> None:
        """Refresh the cached flag after load_state_dict changed the buffer."""
        self._is_active = bool(self.active.item())

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = self.base(hidden_states, *args, **kwargs)
        if not self._is_active:
            return out
        return out + self.adapter(hidden_states).to(out.dtype)


class GrowthManager:
    """Tracks the slots in activation order."""

    def __init__(self, slots: list[GrowableMLP]) -> None:
        self._slots = list(slots)

    @property
    def capacity(self) -> int:
        return len(self._slots)

    def active_count(self) -> int:
        return sum(1 for s in self._slots if s.is_active)

    def activate_next(self) -> int | None:
        for index, slot in enumerate(self._slots):
            if not slot.is_active:
                slot.activate()
                return index
        return None

    def active_parameters(self) -> list[nn.Parameter]:
        return [p for s in self._slots if s.is_active for p in s.adapter.parameters()]

    def adapter_parameters(self) -> list[nn.Parameter]:
        return [p for s in self._slots for p in s.adapter.parameters()]

    def sync(self) -> None:
        for slot in self._slots:
            slot.sync()

    def summary(self) -> list[bool]:
        return [s.is_active for s in self._slots]


def find_blocks(model: nn.Module) -> list[nn.Module]:
    for path in _BLOCK_PATHS:
        obj: Any = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        blocks = list(obj)
        if blocks and all(hasattr(b, "mlp") for b in blocks):
            return blocks
    raise ValueError(
        "Unsupported model architecture: could not find transformer blocks with an .mlp "
        f"under any of {_BLOCK_PATHS}"
    )


def install_growth(model: nn.Module, cfg: GrowthConfig) -> GrowthManager:
    """Wrap every block's MLP with an inactive growth slot. Call before moving the model to a device."""
    hidden = int(model.config.hidden_size)
    blocks = find_blocks(model)
    slots = []
    for block in blocks:
        if isinstance(block.mlp, GrowableMLP):
            raise ValueError("Growth modules are already installed on this model")
        wrapper = GrowableMLP(block.mlp, hidden, cfg.bottleneck)
        block.mlp = wrapper
        slots.append(wrapper)
    ordered = list(reversed(slots)) if cfg.order == "top_down" else slots
    return GrowthManager(ordered)
