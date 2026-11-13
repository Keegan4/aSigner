"""Measure what the residual temporal convolution adds.

    python ablate.py --seeds 0 1 2 3 4

Trains the model with and without the block on every seed. Both runs of a
seed share the same split and the same augmented clips, so the paired
difference shows the block's effect and the spread across seeds shows how much
accuracy moves by chance. Models go to a scratch folder: the served model is
never replaced. Writes ablation.json under runs/ablation-<time>/.

With clips from one signer only, the accuracies are optimistic and the
comparison says little about new signers. Record a second signer first.
"""
import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from train import train

VARIANTS = {"with temporal conv": True, "without temporal conv": False}


def _summary(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "runs": values,
    }


def ablate(clips_dir: Path, seeds: list[int], epochs: int, test_signer: str | None, runs: Path) -> dict:
    folder = runs / time.strftime("ablation-%Y%m%d-%H%M%S")
    scores = {name: {"top1": [], "top5": []} for name in VARIANTS}
    split = None
    with tempfile.TemporaryDirectory() as scratch:
        for seed in seeds:
            for name, temporal_conv in VARIANTS.items():
                print(f"\n=== seed {seed}, {name} ===")
                report = train(clips_dir, Path(scratch) / "signs.onnx", epochs, seed, test_signer,
                               folder / name.replace(" ", "-"), temporal_conv=temporal_conv)
                if "top1" not in report:
                    raise SystemExit("No held-out clips, so there is nothing to compare. Record more clips.")
                scores[name]["top1"].append(report["top1"])
                scores[name]["top5"].append(report["top5"])
                split = report["split"]

    with_, without = (scores[name] for name in VARIANTS)
    result = {
        "seeds": seeds,
        "epochs": epochs,
        "split": split,
        "variants": {name: {k: _summary(v) for k, v in s.items()} for name, s in scores.items()},
        # Positive: the block helps. Paired by seed, so split and augmentation cancel out.
        "difference": {
            k: _summary([a - b for a, b in zip(with_[k], without[k])]) for k in ("top1", "top5")
        },
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ablation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nTemporal conv ablation, {len(seeds)} seeds, {epochs} epochs ({split})")
    print(f"{'':24}{'top-1':>18}{'top-5':>18}")
    for name, summary in result["variants"].items():
        print(f"{name:24}" + "".join(f"{summary[k]['mean']:>11.1%} ± {summary[k]['sd']:.1%}" for k in ("top1", "top5")))
    diff = result["difference"]
    print(f"{'difference (with - w/o)':24}"
          + "".join(f"{diff[k]['mean']:>+11.1%} ± {diff[k]['sd']:.1%}" for k in ("top1", "top5")))
    if len(seeds) > 1 and abs(diff["top1"]["mean"]) <= diff["top1"]["sd"]:
        print("The top-1 difference is within the spread across seeds: no clear effect.")
    print(f"report: {folder / 'ablation.json'}")
    return result


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=Path, default=here.parent / "server" / "data" / "clips")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--test-signer", help="signer id to hold out (default: the one with fewest clips)")
    parser.add_argument("--runs", type=Path, default=here / "runs")
    args = parser.parse_args()
    ablate(args.clips, args.seeds, args.epochs, args.test_signer, args.runs)
