# Track A: Sharanga Slurm Prototype (Single-Node) Runbook

This track runs an OpenAI-compatible HTTP server for `zai-org/GLM-4.7-Flash` **only inside a Slurm allocation** and accesses it via an SSH tunnel.

## Files

- Serve job: `track-a/slurm/glm47_track_a.sbatch`
- Preflight-only job: `track-a/slurm/preflight_only.sbatch`
- Preflight checks: `track-a/bin/preflight.sh`
- Server runner: `track-a/bin/run_server.sh`
- Conda bootstrap (create/reuse env + install requirements): `track-a/bin/bootstrap_conda_env.sh`
- Requirements: `track-a/requirements.txt`
- Laptop smoke test: `track-a/bin/smoke_test.sh`

## Cluster Reality Notes (From Your Session)

- GPU partitions are not named `gpu`. You have partitions like:
  - `gpu_v100_1`, `gpu_v100_2`, `gpu_a100_8`, `gpu_h100_4`
- QoS policy may enforce minimum CPUs for interactive shells (example you saw: `cpulimit` with `MinCPUs=4`).
- Cluster banner requests **package installation (anaconda3/conda/pip)** be done on an **interactive Slurm shell**, not in long batch jobs.
- Default system python on GPU nodes can be old (you saw Python 3.6.8), so you need your own env.

## 0) Get This Repo Onto HPC

You must have the repo on the HPC filesystem, e.g.:

```bash
/home/<user>/dop-project2/track-a/...
```

If `track-a/bin/preflight.sh` is missing on HPC, Slurm jobs will fail with "No such file or directory".

## 1) Pick A GPU Partition

See partitions and time limits:

```bash
sinfo -o "%P %a %l %D %t"
```

Submit jobs with `-p` to choose the GPU type you want:

```bash
sbatch -p gpu_h100_4 ...
```

## 2) (Optional) Preflight Only (VRAM + GPU Visibility)

From the repo root on the login node:

```bash
cd /home/<user>/dop-project2
SKIP_VLLM_CHECK=1 sbatch -p gpu_h100_4 track-a/slurm/preflight_only.sbatch
```

Watch logs:

```bash
tail -f preflight-<jobid>.out
tail -f preflight-<jobid>.err
```

## 3) Create The Python/vLLM Environment (Interactive Slurm Shell)

### 3.1 Get An Interactive Shell (Compute Partition)

Use at least 4 CPUs to satisfy typical interactive QoS minimums:

```bash
cd /home/<user>/dop-project2
srun --export=ALL -p compute --qos=cpulimit -N 1 -n 1 -c 4 --mem=8G -t 0-00:30 --pty bash -i
```

### 3.2 Make Conda Available

If you installed Miniconda to `$HOME/miniconda3`:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda --version
```

If your site provides a module instead, use that (examples only):

```bash
module avail
module load anaconda3
conda --version
```

### 3.3 Accept Anaconda Channel ToS (Once)

If conda blocks non-interactive installs with a ToS error, accept once:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 3.4 Create/Reuse The Env And Install Requirements

Important: `$SCRATCH` may be empty in some contexts. This runbook defaults the env to `$HOME/.conda_envs/...` for reliability.
If you want the env on scratch, set `ENV_PREFIX` explicitly to your real scratch path and use the same value everywhere.

vLLM installation:
- If pip finds a compatible prebuilt wheel, it will install quickly.
- If pip cannot find a wheel, it will try to compile vLLM and you must have a CUDA toolkit with `nvcc` available.
  Load the CUDA module *before* running the bootstrap (example names vary by cluster):

```bash
module avail cuda
module load cuda
which nvcc
nvcc --version
```

```bash
export ENV_PREFIX="$HOME/.conda_envs/glm47-vllm-py310"
export REQUIREMENTS_FILE="$PWD/track-a/requirements.txt"

./track-a/bin/bootstrap_conda_env.sh
```

If pip downloads are blocked, pre-download wheels somewhere and point at them:

```bash
export WHEELHOUSE_DIR=/path/to/wheels
./track-a/bin/bootstrap_conda_env.sh
```

Exit the interactive shell:

```bash
exit
```

Important: submit `sbatch` jobs from the login node (outside the interactive `srun` shell).
Submitting `sbatch` from inside an interactive allocation can leak `SLURM_*` variables into the new job
and cause errors like:
`srun: fatal: cpus-per-task set by two different environment variables ...`

## 4) Submit The Serving Job

From the repo root on the login node:

```bash
cd /home/<user>/dop-project2
ENV_PREFIX="$HOME/.conda_envs/glm47-vllm-py310" \
  sbatch -p gpu_h100_4 track-a/slurm/glm47_track_a.sbatch
```

Monitor:

```bash
squeue -u "$USER"
squeue -j <jobid> -o "%.18i %.9P %.12j %.8T %.10M %.6D %R"
```

Logs are written in the directory you ran `sbatch` from:

```bash
tail -f track-a-<jobid>.out
tail -f track-a-<jobid>.err
```

If you see errors about CPU binding (e.g. "Unable to satisfy cpu bind request"), the sbatch scripts
already force `--cpu-bind=none` for their `srun` steps.

## 5) Find The Node And Tunnel From Your Laptop

Get node:

```bash
squeue -j <jobid> -h -o %N
```

Tunnel (from your laptop):

```bash
ssh -L 8000:<assigned-node>:8000 <username>@hpc.bits-hyderabad.ac.in
```

## 6) Smoke Test (From Your Laptop)

```bash
LLM_BASE_URL=http://localhost:8000 \
LLM_MODEL=glm47-flash30b \
  /path/to/dop-project2/track-a/bin/smoke_test.sh
```

Or curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm47-flash30b",
    "messages": [{"role":"user","content":"Return OK only."}],
    "temperature": 0
  }'
```

## Common Issues

### "invalid partition specified"

Use `sinfo` to find the right partition name and submit with `-p`, e.g. `-p gpu_h100_4`.

### "No such file or directory" pointing to `/var/spool/...`

Submit from the repo root (`cd /home/<user>/dop-project2`) so `SLURM_SUBMIT_DIR` points at the repo.

### "conda: command not found"

In the interactive shell, you must `source "$HOME/miniconda3/etc/profile.d/conda.sh"` (or load your conda module).

### Conda ToS blocking creates

Run the two `conda tos accept ...` commands once, then retry.

### `$SCRATCH` is empty -> paths become `/.conda_envs/...`

Use `${SCRATCH:-$HOME}` as shown above, or set `SCRATCH` to your real scratch path if your site uses one.

### 24h runtime limit

The serve job uses `0-23:50` to stay under 24h; plan to resubmit daily.
