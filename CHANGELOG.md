# Changelog

## 0.1.0 — 2026-10-30

First public cut. Scope is deliberately narrow: **know what your fleet can
train**, and prove it with a real run.

### Added
- `mesh scan` — probe a machine or fleet and report the largest model it can
  actually train, in full precision and with QLoRA.
- `mesh plan` — placement, sync interval, throughput and ETA for a specific
  job on a specific fleet.
- `mesh up` — coordinator + dashboard in one command, with `--demo` for a
  synthetic fleet.
- `mesh agent` — the installable client: probes the device and heartbeats.
- `mesh serve` — HTTP API (`/api/v1/stats`, `/nodes`, `/capacity`, `/heartbeat`).
- Fleet dashboard: capacity ladder, geography, growth, hardware mix, inventory.
- Reference training runtime: DiLoCo outer loop, int8 pseudo-gradient
  compression, robust aggregation, churn-tolerant round barrier.
- Docker image and compose file.

### Known gaps
- No pipeline parallelism: a model must fit on one node.
- Coordinator is a single trusted server; no signed node identity.
- Verification is a gate (norm + consensus direction), not a proof.
- Transport is plain TCP with a shared-secret header.
