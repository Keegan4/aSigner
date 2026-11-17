"""Measure what one block of the model adds.

    python ablate.py --seeds 0 1 2 3 4
    python ablate.py --block graph-conv --seeds 0 1 2 3 4

The block is the residual temporal convolution unless --block says otherwise.
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

# Each block that can be measured: train()'s keyword for it, and its name in the report.
BLOCKS = {
    "temporal-conv": ("temporal_conv", "temporal conv"),
    "graph-conv": ("graph_conv", "graph conv"),
}


def _summary(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "runs": values,
    }


def ablate(clips_dir: Path, seeds: list[int], epochs: int, test_signer: str | None, runs: Path,
           block: str = "temporal-conv") -> dict:
    keyword, block_name = BLOCKS[block]
    variants = {f"with {block_name}": True, f"without {block_name}": False}
    folder = runs / time.strftime("ablation-%Y%m%d-%H%M%S")
    scores = {name: {"top1": [], "top5": []} for name in variants}
    split = None
    with tempfile.TemporaryDirectory() as scratch:
        for seed in seeds:
            for name, included in variants.items():
                print(f"\n=== seed {seed}, {name} ===")
                report = train(clips_dir, Path(scratch) / "signs.onnx", epochs, seed, test_signer,
                               folder / name.replace(" ", "-"), **{keyword: included})
                if "top1" not in report:
                    raise SystemExit("No held-out clips, so there is nothing to compare. Record more clips.")
                scores[name]["top1"].append(report["top1"])
                scores[name]["top5"].append(report["top5"])
                split = report["split"]

    with_, without = (scores[name] for name in variants)
    result = {
        "block": block,
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

    print(f"\n{block_name.capitalize()} ablation, {len(seeds)} seeds, {epochs} epochs ({split})")
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
    parser.add_argument("--block", choices=list(BLOCKS), default="temporal-conv",
                        help="the block to train with and without (default: temporal-conv)")
    args = parser.parse_args()
    ablate(args.clips, args.seeds, args.epochs, args.test_signer, args.runs, args.block)
