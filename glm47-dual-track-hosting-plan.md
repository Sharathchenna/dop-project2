# Dual-Track LLM Hosting Plan (Sharanga Prototype + No-Constraint Production)

## Summary
Build one document with two deployment tracks for `GLM-4.7-Flash-30B`:
1. `Track A` Sharanga-constrained prototype (Slurm, single GPU node, 24-hour job cap).
2. `Track B` unconstrained production deployment (multi-node, always-on, public HTTPS API).
3. A migration section that keeps client API unchanged so app code does not change between tracks.

The access model will be:
1. Prototype: SSH tunnel to the running Slurm job.
2. Production: HTTPS endpoint with API key.

## Constraints Locked From FAQ
1. Only one GPU node is available (FAQ pages 32 and 34).
2. GPU jobs with `-t > 24h` are terminated (FAQ pages 32 and 35).
3. No running code on login node; jobs must go through Slurm (FAQ page 37).
4. Use `srun` under Slurm allocation (FAQ pages 31 and 33).
5. `$SCRATCH` retention is 15 days and not backed up (FAQ pages 15 and 16).

## Document Structure to Produce
1. `Section A`: Architecture overview (dual-track + migration).
2. `Section B`: Track A Sharanga prototype runbook.
3. `Section C`: Track B production runbook.
4. `Section D`: Access instructions (exact commands for both tracks).
5. `Section E`: Validation checklist and acceptance tests.
6. `Section F`: Risks, assumptions, rollback.

## Track A (Sharanga Prototype) Implementation Spec
1. Submit as Slurm GPU job only:
- `#SBATCH -p gpu`
- `#SBATCH -N 1`
- `#SBATCH --gres=gpu:<count>`
- `#SBATCH -t 0-23:50` (hard cap below 24h)
- launch server with `srun` only.

2. Add preflight gate before final launch:
- detect GPU count and VRAM on allocated node.
- if resources satisfy 30B serving requirement, launch `GLM-4.7-Flash-30B`.
- if not, mark Track A as “integration-only” and route tests to Track B (no model downgrade by default).

3. Runtime behavior:
- bind service to compute node port (example `8000`).
- write logs to Slurm output/error files.
- store model/cache in persistent location (not volatile scratch unless explicitly intended).

4. Restart policy:
- prototype is scheduled, not always-on.
- document daily resubmission pattern due to 24h limit.

## Track B (No-Constraint Production) Implementation Spec
1. Use multi-node serving stack (vLLM + Ray or equivalent).
2. Expose a stable HTTPS API behind load balancer/ingress.
3. Enable auth (`Bearer` token), rate limits, metrics, and centralized logs.
4. Keep model alias constant (example `glm47-flash30b`) for client stability.
5. Define autoscaling and rolling restart strategy.

## Access: How You Will Use It
1. Prototype access (Sharanga):
- submit job, then get assigned node: `squeue -j <jobid> -h -o %N`
- create tunnel from your laptop:
  - `ssh -L 8000:<assigned-node>:8000 <username>@hpc.bits-hyderabad.ac.in`
- call endpoint locally:
  - `http://localhost:8000/v1/chat/completions`

2. Production access:
- call public/internal HTTPS endpoint:
  - `https://<your-domain>/v1/chat/completions`
- include `Authorization: Bearer <API_KEY>`

3. Client request format is identical across both tracks (OpenAI-compatible), so only `base_url` and credentials change.

## Public APIs / Interfaces / Types
1. API path: `/v1/chat/completions`
2. Model field: `model: "glm47-flash30b"`
3. Auth:
- Track A: none (tunnel-protected).
- Track B: bearer token required.
4. Health and metrics:
- `/health` (or equivalent)
- `/metrics` for Prometheus.

## Test Cases and Scenarios
1. Track A smoke test via SSH tunnel returns valid completion.
2. Track A tunnel reconnect works after SSH drop.
3. Track A job termination at walltime is handled by restart runbook.
4. Track B external HTTPS call succeeds with valid API key.
5. Track B rejects unauthorized requests.
6. Same client payload works unchanged across Track A and Track B.
7. Migration test: switch `base_url` from localhost tunnel to production URL without code changes.

## Migration Plan (Prototype -> Production)
1. Freeze client contract during prototype (`model` alias + endpoint shape).
2. Deploy Track B with same contract.
3. Run dual validation against both endpoints.
4. Cut over clients by changing only environment variables:
- `LLM_BASE_URL`
- `LLM_API_KEY`
5. Keep Track A as fallback testbed.

## Assumptions and Defaults
1. Source of cluster policy is `/Users/sharathchenna/Downloads/hpc_sharanga_faq.pdf` (created November 6, 2021), so live limits must be rechecked with `scontrol show partition` before execution.
2. Default is no public exposure from Sharanga prototype; access is through SSH tunnel.
3. No model-size downgrade unless you explicitly approve it.
4. Production is the only always-on/public endpoint.
