"""Saving pretrained weights and starting a new model from them."""
from pathlib import Path

import torch
from torch import nn


def save_checkpoint(model: nn.Module, labels: list[str], source: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "labels": labels, "source": source}, path)


def load_into(model: nn.Module, labels: list[str], path: Path) -> str:
    """Start `model` from the checkpoint at `path` and return where its data came from.

    Everything but the last layer is copied as it is. The last layer has one
    row per label, so rows are copied by label name: a label the checkpoint
    never saw keeps its fresh initialisation.
    """
    # weights_only: a checkpoint is a file from disk and must not run code on load.
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state, own = saved["state"], model.state_dict()
    last = max((k for k in own if k.endswith(".weight") and own[k].ndim == 2), key=_position(own))
    last_bias = last.removesuffix("weight") + "bias"
    for key, value in state.items():
        if key in (last, last_bias):
            continue
        if key not in own or own[key].shape != value.shape:
            raise SystemExit(
                f"{path} does not fit this model ({key}). It was made for a different "
                "feature layout or network; run pretrain.py again."
            )
        own[key] = value
    for row, label in enumerate(saved["labels"]):
        if label in labels:
            own[last][labels.index(label)] = state[last][row]
            own[last_bias][labels.index(label)] = state[last_bias][row]
    model.load_state_dict(own)
    return str(saved["source"])


def _position(state: dict):
    order = list(state)
    return order.index
