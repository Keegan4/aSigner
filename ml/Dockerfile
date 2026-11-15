# Training image: pulls takes from the training page's database and trains on an
# NVIDIA GPU. Built for a GTX 1660 Super (Turing, compute capability 7.5); any
# card PyTorch's CUDA build supports will do. Without a GPU it trains on the CPU,
# or refuses to when REQUIRE_GPU=1 (the default in deploy/trainer).
#
# Build from the repository root, since ml/ imports server/app/recognition:
#   docker build -f ml/Dockerfile -t asigner-trainer .
FROM python:3.13-slim

# PyTorch's CUDA wheels carry their own CUDA runtime, so only the host's NVIDIA
# driver and the NVIDIA Container Toolkit are needed, not a CUDA base image.
ARG TORCH_CUDA=cu128
ARG CUDA_ARCH=sm_75

WORKDIR /work
COPY ml/requirements.txt ml/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
 && pip install --no-cache-dir -r ml/requirements.txt
# Fail the build if this PyTorch has no kernels for the target card. get_arch_list()
# is empty without a GPU, so read the flags it was compiled with instead.
RUN python -c "import sys, torch; archs = torch._C._cuda_getArchFlags().split(); print('torch', torch.__version__, 'CUDA', torch.version.cuda, archs); sys.exit(0 if '${CUDA_ARCH}' in archs else 'This PyTorch build has no ${CUDA_ARCH} kernels; pick another TORCH_CUDA')"

# Only the shared feature code from the server: the normaliser and the ONNX loader.
COPY server/app/__init__.py server/app/__init__.py
COPY server/app/recognition server/app/recognition
COPY ml ml

WORKDIR /work/ml
ENTRYPOINT ["sh", "docker-train.sh"]
