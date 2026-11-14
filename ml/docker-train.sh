#!/bin/sh
# The training container's entry point: pull new takes, then train and export.
# Extra arguments go to train.py, for example --epochs 120 or --test-signer t0123456789abcdef.
set -e

DATA=${DATA_DIR:-/data}
OUT=${OUT_DIR:-/out}

python - <<'EOF'
import os, sys, torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
elif os.environ.get("REQUIRE_GPU", "1") == "1":
    sys.exit("No GPU visible to PyTorch. Check the NVIDIA driver and the NVIDIA Container "
             "Toolkit (docker run --rm --gpus all ubuntu nvidia-smi), or set REQUIRE_GPU=0 "
             "to train on the CPU.")
else:
    print("No GPU visible: training on the CPU (REQUIRE_GPU=0).")
EOF

python pull_samples.py --out "$DATA/collected"

# Clips recorded on /record, if a folder of them is mounted at /data/recorded.
CLIPS="$DATA/collected"
if [ -d "$DATA/recorded" ] && [ -n "$(ls -A "$DATA/recorded" 2>/dev/null)" ]; then
    CLIPS="$CLIPS $DATA/recorded"
fi

mkdir -p "$OUT/runs"
# shellcheck disable=SC2086  # CLIPS is a list of folders
exec python train.py --clips $CLIPS --out "$OUT/signs.onnx" --runs "$OUT/runs" "$@"
