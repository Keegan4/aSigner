# Training-data collector on a Linux VM

Runs the training page (`/train`) on a separate machine so other people can record signs, and add signs of their own, into a Postgres database. You pull the labelled takes from that database to train on.

What is stored, in the `training_samples` table: the landmark frames, the label, the frame count and rate, the time of the take, and an anonymous contributor id (a hash of a random token the browser makes up). There is no account, name, email or video. The contributor id lets `train.py` hold out a whole person when it measures accuracy, and lets people undo their own last take.

Signs people add go in `training_signs` (name and an optional "how it is made" note). Lesson signs and "No sign" are always on the list.

## Set up the VM

Any Linux VM with about 2 GB of RAM (the web build needs it), Docker with the Compose plugin, and internet access for the build.

```bash
git clone <this repository> asigner
cd asigner/deploy/training-vm
cp .env.example .env
nano .env        # fill in SITE_ADDRESS, POSTGRES_PASSWORD, SESSION_SECRET, TRAINING_ACCESS_CODE
docker compose up -d --build
```

Open `https://<SITE_ADDRESS>/` (it goes to `/train`), type the access code and record.

**HTTPS is required.** Browsers only allow the camera on HTTPS pages (or on localhost). Caddy handles it:

- With a domain name pointing at the VM and ports 80 and 443 open to the internet, Caddy gets a real certificate automatically.
- With an IP address on your own network, Caddy signs its own certificate. Each browser shows a warning once; accept it to continue. To get rid of the warning, install Caddy's root certificate on the devices: `docker compose cp caddy:/data/caddy/pki/authorities/local/root.crt .`

To update after pulling new code: `git pull && docker compose up -d --build`. The server applies database migrations when it starts.

## Pull the data for training

On your own machine, with the ML environment from the main README:

```bash
ssh -N -L 5433:localhost:5432 you@the-vm
```

Leave that running (it forwards your port 5433 to Postgres on the VM), then in another terminal:

```bash
cd ml
.venv/Scripts/python pull_samples.py --db postgresql://asigner:<POSTGRES_PASSWORD>@localhost:5433/asigner
.venv/Scripts/python train.py --clips ../server/data/clips data/collected
```

`pull_samples.py` writes one clip file per take under `ml/data/collected/<label>/`, in the format `/record` uses. Running it again only fetches new takes, and removes local files for takes deleted since. Leave out `../server/data/clips` to train on collected takes only.

If you set `POSTGRES_BIND=0.0.0.0`, skip the tunnel and use the VM's address and port 5432 instead.

### Reading the table directly

```sql
SELECT label, count(*) FROM training_samples GROUP BY label ORDER BY label;
SELECT id, label, frame_count, fps, contributor, frames FROM training_samples;
```

`frames` is `frame_count × 256` little-endian float32 values back to back, laid out as in `server/app/recognition/features.py`. In Python: `np.frombuffer(frames, "<f4").reshape(frame_count, 256)`.

## Looking after it

- Back up: `docker compose exec postgres pg_dump -U asigner asigner > backup.sql`
- Delete a bad take: `docker compose exec postgres psql -U asigner -c "DELETE FROM training_samples WHERE id = 123"`
- Remove a sign someone added by mistake: on the page, while it has no takes, or with `DELETE FROM training_signs WHERE label = 'TYPO'` (and its takes, if any, from `training_samples`).
- Change the access code: edit `.env`, then `docker compose up -d`.

Anyone with the access code can add takes and signs. There is no per-person limit, so share the code only with people you trust to record.
