"""Teach the sign model about hands and bodies without labels, before pretrain.py.

    python pretrain_masked.py --epochs 60 --out runs/masked.pt
    python pretrain.py --init runs/masked.pt
    python train.py --init runs/pretrained.pt

Stretches of a clip are hidden, a hand for a few frames here, the body for a
few there, and the model has to fill in where the hidden landmarks were from
the frames and body parts it can still see. No labels are used, so the 2,000
WLASL words with about five clips each count as 10,000 clips of signing
instead of 2,000 nearly empty classes. The hidden parts are presented exactly
as the tracker presents a part it lost (zeros, with the part marked absent).

It reads the same public data as pretrain.py and holds out the same clips,
augmented the same way (whole clips and live translator windows).
runs/masked.json has the error on those for every epoch, next to the error of
guessing each hidden part's average visible position in that clip: the model
has learned movement only where it beats that. The checkpoint holds everything
but a classifier; pretrain.py --init and train.py --init both start from it.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from checkpoint import save_checkpoint
from data import Clip, augmented_samples
from pretrain import load_public
from train import SignNet, model_inputs, pick_device

from app.recognition.features import COORDINATE_SIZE, NORMALISED_PARTS, REGION_PARTS  # path set up by data.py

SOURCE_NOTE = "masked landmarks, no labels"
HIDDEN_SPANS = 3  # stretches hidden per region in each clip
SPAN_FRAMES = (3, 6)  # shortest and longest stretch; three of them hide about 35% of a region's frames


def hide(x: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Model inputs with stretches of each region hidden, and which coordinates to predict.

    `x` is (batch, time, NORMALISED_SIZE). Each region (left hand, right hand,
    body, face) loses its own stretches, so the others are there to go by.
    The second result is 1.0 for every coordinate that was hidden and had been
    tracked: a part the tracker never saw has no right answer to learn from.
    """
    batch, time, _ = x.shape
    shape = (batch, len(REGION_PARTS), HIDDEN_SPANS)
    length = torch.randint(SPAN_FRAMES[0], SPAN_FRAMES[1] + 1, shape, generator=generator)
    start = torch.randint(0, time, shape, generator=generator)
    frame = torch.arange(time).view(1, 1, 1, time)
    hidden = ((frame >= start[..., None]) & (frame < (start + length)[..., None])).any(dim=2)  # (batch, region, time)
    hidden = hidden.to(x.device)

    shown, wanted = x.clone(), torch.zeros_like(x[..., :COORDINATE_SIZE])
    for region, parts in enumerate(REGION_PARTS):
        gone = hidden[:, region, :, None]
        for part in parts:
            flag = COORDINATE_SIZE + part
            wanted[..., NORMALISED_PARTS[part]] = (gone & (x[..., flag:flag + 1] > 0.5)).to(x.dtype)
            shown[..., NORMALISED_PARTS[part]] = shown[..., NORMALISED_PARTS[part]].masked_fill(gone, 0.0)
            shown[..., flag:flag + 1] = shown[..., flag:flag + 1].masked_fill(gone, 0.0)
    return shown, wanted


def hidden_error(predicted: torch.Tensor, x: torch.Tensor, wanted: torch.Tensor) -> torch.Tensor:
    """Mean error over the hidden, tracked coordinates only."""
    error = nn.functional.smooth_l1_loss(predicted, x[..., :COORDINATE_SIZE], reduction="none", beta=0.1)
    return (error * wanted).sum() / wanted.sum().clamp(min=1.0)


def average_guess(shown: torch.Tensor) -> torch.Tensor:
    """The yardstick: each part's average position over the frames of the clip where it is still visible."""
    guess = torch.zeros_like(shown[..., :COORDINATE_SIZE])
    for part, columns in enumerate(NORMALISED_PARTS):
        visible = (shown[..., COORDINATE_SIZE + part:COORDINATE_SIZE + part + 1] > 0.5).to(shown.dtype)
        guess[..., columns] = ((shown[..., columns] * visible).sum(dim=1, keepdim=True)
                               / visible.sum(dim=1, keepdim=True).clamp(min=1.0))
    return guess


class Filler(nn.Module):
    """The sign model's encoder with a small head that names every frame's landmark positions."""

    def __init__(self, net: SignNet, width: int):
        super().__init__()
        self.net = net
        self.decoder = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, COORDINATE_SIZE))

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.net.encode(clip)[:, 1:])  # token 0 is the summary token


def evaluate(filler: Filler, inputs: dict[str, np.ndarray], device: torch.device, seed: int) -> dict:
    """Error on held-out clips with the same parts hidden every time, and the average guess's error on them."""
    generator = torch.Generator().manual_seed(seed)
    filler.eval()
    out = {}
    with torch.no_grad():
        for name, clips in inputs.items():
            x = torch.from_numpy(clips).to(device)
            shown, wanted = hide(x, generator)
            out[name] = {"loss": float(hidden_error(filler(shown), x, wanted)),
                         "average_guess": float(hidden_error(average_guess(shown), x, wanted)),
                         "samples": len(clips)}
    return out


def pretrain_masked(data: Path, out: Path, epochs: int, seed: int, casl: Path | None = None,
                    wlasl: Path | None = None, warmup: int | None = None, asl_signs: Path | None = None,
                    per_sign: int | None = 100) -> dict:
    if warmup is not None and not 0 < warmup < epochs:
        raise SystemExit(f"--warmup must be between 1 and {epochs - 1} for {epochs} epochs.")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    train_clips, held_out, source, _ = load_public(data, casl, wlasl, rng, asl_signs, per_sign)
    held_out = {name: group for name, group in held_out.items() if group}
    print(f"{len(train_clips)} clips to learn from, {sum(map(len, held_out.values()))} held out, labels unused ({source})")

    device = pick_device()
    width = 320
    filler = Filler(SignNet(1, width), width).to(device)
    print(f"training on {device}")
    steps = (len(train_clips) + 63) // 64
    optimiser = torch.optim.AdamW(filler.parameters(), lr=1e-3, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=1e-3, epochs=epochs, steps_per_epoch=steps,
                                                   pct_start=warmup / epochs if warmup else 0.3)
    held_out_inputs = {name: model_inputs(group) for name, group in held_out.items()}

    history = []
    for epoch in range(1, epochs + 1):
        seeds = rng.integers(np.iinfo(np.int64).max, size=len(train_clips))
        x_all = torch.from_numpy(np.stack([x for x, _ in augmented_samples(train_clips, seeds)]))
        filler.train()
        total = 0.0
        for batch in torch.from_numpy(rng.permutation(len(train_clips))).tensor_split(steps):
            x = x_all[batch].to(device)
            shown, wanted = hide(x, generator)
            optimiser.zero_grad()
            loss = hidden_error(filler(shown), x, wanted)
            loss.backward()
            optimiser.step()
            schedule.step()
            total += loss.item() * len(batch)
        row = {"epoch": epoch, "train": {"loss": total / len(train_clips)}} | evaluate(filler, held_out_inputs, device, seed)
        history.append(row)
        print(f"epoch {epoch:3d}  loss {row['train']['loss']:.4f}  "
              + "  ".join(f"{name} {row[name]['loss']:.4f} (average guess {row[name]['average_guess']:.4f})"
                          for name in held_out_inputs))

    source = f"{SOURCE_NOTE}: {source}"
    # No labels: train.py and pretrain.py give every label a fresh output row.
    save_checkpoint(filler.net, [], source, out)
    report = {"samples": len(train_clips), "epochs": epochs, "seed": seed, "source": source, "history": history}
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(f"checkpoint: {out}\nhistory: {out.with_suffix('.json')}")
    return report


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=here / "data" / "public" / "asl-now-fingerspelling")
    parser.add_argument("--out", type=Path, default=here / "runs" / "masked.pt")
    parser.add_argument("--casl", type=Path, default=here / "data" / "public" / "casl-w60",
                        help="CASL-W60 parquet files; used when present")
    parser.add_argument("--wlasl", type=Path, default=here / "data" / "public" / "wlasl-landmarks",
                        help="WLASL landmark parquet files (academic, non-commercial); used when present")
    parser.add_argument("--asl-signs", type=Path, default=here / "data" / "public" / "asl-signs",
                        help="Kaggle's asl-signs folder (the one holding train.csv); used when present")
    parser.add_argument("--per-sign", type=int, default=100,
                        help="asl-signs clips to read for each sign (default 100; 0 for all of them)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup", type=int,
                        help="epochs the learning rate rises for before it falls (default: 30%% of --epochs)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    pretrain_masked(args.data, args.out, args.epochs, args.seed, args.casl, args.wlasl, args.warmup,
                    args.asl_signs, args.per_sign or None)
