# Training-data collector on a Linux VM

Runs the training page (`/train`) on a separate machine so other people can record signs, and add signs of their own, into a Postgres database. You pull the labelled takes from that database to train on.

What is stored, in the `training_samples` table: the landmark frames, the label, the frame count and rate, the time of the take, and an anonymous contributor id (a hash of a random token the browser makes up). There is no account, name, email or video. The contributor id lets `train.py` hold out a whole person when it measures accuracy, and lets people undo their own last take.

Signs people add go in `training_signs` (name and an optional "how it is made" note). Lesson signs and "No sign" are always on the list.

```
Contributor's browser ──HTTPS──▶ Caddy ─▶ web (/train)
   (or via Cloudflare)              └──▶ server (/api) ─▶ Postgres
                                                           ▲
Your PC / GPU machine ──SSH tunnel─────────────────────────┘  pull_samples.py, deploy/trainer
```

## 1. Make the VM

- Ubuntu Server 22.04 or 24.04, 2 CPUs, at least 2 GB of RAM (the web build needs it), 20 GB of disk.
- Network: **bridged** (not NAT), so the VM gets its own address on your network that other machines can reach. Note it with `ip -4 addr` (for example `192.168.1.50`); a DHCP reservation on your router keeps it fixed.
- Enable SSH (`sudo apt install openssh-server`), so you can manage it and pull data from your PC.

## 2. Install Docker

```bash
sudo apt-get update && sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER   # then log out and back in
```

## 3. Configure and start

```bash
git clone https://github.com/Keegan4/aSigner.git asigner
cd asigner/deploy/training-vm
cp .env.example .env
openssl rand -hex 24   # use for POSTGRES_PASSWORD
openssl rand -hex 32   # use for SESSION_SECRET
nano .env              # SITE_ADDRESS (the VM's address), the two above, TRAINING_ACCESS_CODE
docker compose up -d --build
docker compose ps      # postgres healthy; server, web, caddy up
```

The first build takes several minutes. The server applies database migrations when it starts, and every container restarts by itself when the VM reboots.

Open `https://<SITE_ADDRESS>/` from a machine on the same network (it goes to `/train`), type the access code and record.

If the VM already runs a Postgres on port 5432, set `POSTGRES_PORT` in `.env` to another port.

## 4. Let people reach it

Browsers only give a page the camera over HTTPS with a certificate they trust, so how people reach the VM decides how much setup they need. Pick one.

| | Who can reach it | Certificate | What it takes |
|---|---|---|---|
| **A. Cloudflare tunnel, own account** (recommended) | Anyone, anywhere | Real, nothing to install | Free Cloudflare account and a domain on it |
| B. Quick tunnel | Anyone with the link | Real | Nothing; the link changes on every restart |
| C. The VM's address | Your network only | Self-signed | Each device trusts `caddy-root.crt` once |

Tunnels connect out from the VM, so no ports are opened on your router. They send plain HTTP to Caddy's port 8080 inside Docker's network, which is not published on the VM. Traffic passes through Cloudflare: landmark numbers and labels, never video.

### A. Cloudflare tunnel with a free account (fixed address)

1. Sign up at [dash.cloudflare.com](https://dash.cloudflare.com) (the Free plan is enough).
2. Add a domain to the account. A tunnel's public address must be on a domain Cloudflare manages: buy one through Cloudflare Registrar, or add a domain you own and switch its nameservers to the ones Cloudflare shows.
3. Open **Zero Trust** (first time: choose a team name and the Free plan), then **Networks → Tunnels → Create a tunnel → Cloudflared**. Name it, for example `asigner-training`.
4. On the install page, copy only the token: the long string after `--token` in any of the commands shown. Do not run Cloudflare's install command; Docker runs cloudflared for you.
5. Add a **public hostname**: subdomain `train`, your domain, service type **HTTP**, URL **`caddy:8080`**. Save.
6. On the VM:

   ```bash
   nano .env    # CLOUDFLARE_TUNNEL_TOKEN=<the token>
   docker compose --profile tunnel up -d
   docker compose logs tunnel | grep -i "registered tunnel connection"
   ```

7. Open `https://train.<your domain>/`. Send that address and the access code to contributors.

The tunnel keeps running across reboots. To stop it: `docker compose --profile tunnel stop tunnel`. Keep the token secret: it lets anyone run your tunnel. If it leaks, refresh it on the tunnel's page in Zero Trust.

Optional: to let in only people you name, add a **Zero Trust → Access → Application** for the hostname with a policy listing their email addresses. Cloudflare then emails them a one-time code before the page opens, on top of the access code.

### B. Quick tunnel (no account, for trying things out)

```bash
docker compose --profile quick-tunnel up -d
docker compose logs quick-tunnel | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com'
```

Anyone with that link reaches the page, and only the access code stops them, so use a long random one. The link changes whenever the container restarts. Stop it with `docker compose --profile quick-tunnel stop quick-tunnel`.

### C. The VM's address on your network

Nothing more to run. Caddy signs its own certificate for the address in `SITE_ADDRESS`, so browsers warn and may refuse the camera until each device trusts it:

```bash
docker compose cp caddy:/data/caddy/pki/authorities/local/root.crt caddy-root.crt
```

Copy `caddy-root.crt` to each device and add it to the trusted root certificates (on Windows: `Import-Certificate -FilePath caddy-root.crt -CertStoreLocation Cert:\CurrentUser\Root`, then restart the browser).

With a domain pointing at your home IP and ports 80 and 443 forwarded to the VM, putting that domain in `SITE_ADDRESS` gets a real certificate from Let's Encrypt instead. That opens your network to the internet, which the tunnel avoids.

## 5. Pull the data for training

On your own machine, with the ML environment from the main README:

```bash
ssh -N -L 5433:localhost:5432 you@the-vm
```

Leave that running (it forwards your port 5433 to Postgres on the VM), then in another terminal:

```bash
cd ml
.venv/Scripts/python pull_samples.py --db postgresql://asigner:<POSTGRES_PASSWORD>@127.0.0.1:5433/asigner
.venv/Scripts/python train.py --clips ../server/data/clips data/collected
```

`pull_samples.py` writes one clip file per take under `ml/data/collected/<label>/`, in the format `/record` uses. Running it again only fetches new takes, and removes local files for takes deleted since. Leave out `../server/data/clips` to train on collected takes only. Use `127.0.0.1` rather than `localhost`: on Windows, `localhost` may try IPv6 first, which the SSH tunnel does not answer.

To pull and train in one step on an NVIDIA GPU, use the training container in [../trainer](../trainer/README.md).

If you set `POSTGRES_BIND=0.0.0.0`, skip the SSH tunnel and use the VM's address and `POSTGRES_PORT` instead. Postgres is then open to your whole network, protected by its password.

### Reading the table directly

```sql
SELECT label, count(*) FROM training_samples GROUP BY label ORDER BY label;
SELECT id, label, frame_count, fps, contributor, frames FROM training_samples;
```

`frames` is `frame_count × 256` little-endian float32 values back to back, laid out as in `server/app/recognition/features.py`. In Python: `np.frombuffer(frames, "<f4").reshape(frame_count, 256)`.

## Looking after it

- Update after new code is merged: `git pull && docker compose up -d --build` (add `--profile tunnel` if you use it).
- Back up: `docker compose exec postgres pg_dump -U asigner asigner > backup.sql`
- Delete a bad take: `docker compose exec postgres psql -U asigner -c "DELETE FROM training_samples WHERE id = 123"`
- Remove a sign someone added by mistake: on the page, while it has no takes, or with `DELETE FROM training_signs WHERE label = 'TYPO'` (and its takes, if any, from `training_samples`).
- Change the access code: edit `.env`, then `docker compose up -d`.

Anyone with the access code can add takes and signs. There is no per-person limit, so share the code only with people you trust to record.
