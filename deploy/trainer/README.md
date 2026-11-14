# Training on the GPU machine

Trains the sign model in a container on the machine with the GPU (a GTX 1660 Super). The container pulls every new take from the training page's Postgres ([../training-vm](../training-vm/README.md)), trains, and writes `out/signs.onnx` plus a report under `out/runs/`. The GPU is used for training only: the server always runs the exported model on its CPU.

The image is built for compute capability 7.5 (Turing: GTX 16xx, RTX 20xx); the build fails if the chosen PyTorch build has no kernels for it.

## Set up the GPU machine once

1. **NVIDIA driver**, 570 or newer for the default `cu128` build. Check with `nvidia-smi`.
2. **Docker**:
   - Linux: Docker Engine ([docs.docker.com/engine/install](https://docs.docker.com/engine/install/)), then the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.
   - Windows: Docker Engine inside WSL 2 works the same way; install the Windows NVIDIA driver only (WSL picks it up), and the toolkit inside WSL. Run the commands below inside WSL.
3. Check that containers can see the card:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

## Train

```bash
cd deploy/trainer
cp .env.example .env      # set TRAINING_DATABASE_URL
ssh -N -L 5433:localhost:5432 you@the-vm &    # unless the VM's Postgres is open to the network
docker compose run --rm --build trainer
```

It prints the GPU it found (or stops if there is none, unless `REQUIRE_GPU=0`), how many takes it pulled, then the usual training output. Arguments after `trainer` go to `train.py`:

```bash
docker compose run --rm trainer --epochs 120 --test-signer t0123456789abcdef
```

Then copy `out/signs.onnx` to `server/models/signs.onnx` wherever the app runs; the server loads it on the next connection.

Pulled takes stay in `data/collected/`, so later runs only fetch new ones. To train on clips recorded on `/record` as well, copy that `server/data/clips` folder to `data/recorded/` here.

## If it does not see the GPU

- `docker run --rm --gpus all ubuntu nvidia-smi` fails: the NVIDIA Container Toolkit is missing or Docker was not restarted after configuring it.
- That works but training says no GPU: the driver is too old for `TORCH_CUDA`. Update the driver, or set `TORCH_CUDA=cu126` in `.env` and run with `--build`.
