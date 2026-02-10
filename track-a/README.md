# Track A (Sharanga Slurm Prototype) Runbook

This track runs the model server *only* as a Slurm GPU job on a single compute node and accesses it via an SSH tunnel.

## What You Get

- Slurm batch script: `track-a/slurm/glm47_track_a.sbatch`
- Preflight checks (GPU count + VRAM + vLLM availability): `track-a/bin/preflight.sh`
- Server launcher (OpenAI-compatible endpoint): `track-a/bin/run_server.sh`
- Smoke test (run on your laptop after tunneling): `track-a/bin/smoke_test.sh`

## Assumptions

- You will run this on Sharanga via Slurm (no model serving on login node).
- Your Python environment on the compute node already has `vllm` installed, or you will load/activate it in the batch script.
- The model is accessible as a HF model id (or local path) via `MODEL_ID`.

## 1) Submit The Job

From the cluster (in this repo directory):

```bash
sbatch track-a/slurm/glm47_track_a.sbatch
```

If your cluster has GPU partitions like `gpu_v100_1`, `gpu_v100_2`, `gpu_a100_8`, `gpu_h100_4`, you can override:

```bash
sbatch -p gpu_h100_4 track-a/slurm/glm47_track_a.sbatch
```

### Conda Env Bootstrapping (vLLM)

On your cluster the default `python` may be too old (you saw Python 3.6.8). The server job script can bootstrap a cached conda env and install `vllm` automatically.

Defaults (override via env vars at submit time):

- `BOOTSTRAP_CONDA_ENV=1` (enabled in `glm47_track_a.sbatch`)
- `PYTHON_VERSION=3.10`
- `ENV_PREFIX=$SCRATCH/.conda_envs/glm47-vllm-py310`
- `REQUIREMENTS_FILE=track-a/requirements.txt`

Example:

```bash
cd /home/<user>/dop-project2
BOOTSTRAP_CONDA_ENV=1 PYTHON_VERSION=3.10 sbatch -p gpu_h100_4 track-a/slurm/glm47_track_a.sbatch
```

If pip downloads are blocked on compute nodes, pre-download wheels to a directory and use:

```bash
WHEELHOUSE_DIR=/path/to/wheels BOOTSTRAP_CONDA_ENV=1 sbatch track-a/slurm/glm47_track_a.sbatch
```

Watch the queue:

```bash
squeue -u "$USER"
```

Get the assigned node once running:

```bash
squeue -j <jobid> -h -o %N
```

## 2) Create The SSH Tunnel (From Your Laptop)

```bash
ssh -L 8000:<assigned-node>:8000 <username>@hpc.bits-hyderabad.ac.in
```

If you changed the port in the job script, use the same local/remote port.

## 3) Call The API Locally (From Your Laptop)

```bash
./track-a/bin/smoke_test.sh
```

Or direct curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm47-flash30b",
    "messages": [{"role":"user","content":"Say hello in one sentence."}],
    "temperature": 0.2
  }'
```

## 4) Logs / Debug

- Slurm output: `track-a-%j.out` (in the submit directory)
- Slurm error: `track-a-%j.err` (in the submit directory)

Common failure modes:

- Preflight exits due to insufficient VRAM: increase resources or adjust `MIN_VRAM_GB_PER_GPU` (only if you are sure).
- `vllm` not found: load modules / activate conda in `track-a/slurm/glm47_track_a.sbatch`.
- Port in use on node: set `PORT` to a different value.

## 5) 24h Walltime Pattern

Sharanga terminates GPU jobs beyond 24h. This job uses `0-23:50`.
Plan to resubmit daily; you can keep the client contract stable by keeping the same API path and model alias.
