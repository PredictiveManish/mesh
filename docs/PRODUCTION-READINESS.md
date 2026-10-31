# From this prototype to a useful product

What exists in this repository is a vertical slice: every plane of the system
is present and working, but each is at roughly 5–10% of production depth. This
document is the gap, subsystem by subsystem, with effort estimates.

---

## 1. What the prototype actually proves

Five claims, each backed by a run in `RESULTS.md`:

1. **The planner's memory arithmetic is right.** A 100-node mixed consumer
   fleet cannot full-fine-tune a 1.2B model, because 16 bytes per parameter
   puts 0.8B on a 16 GB card. Data parallelism scales throughput, not model
   size.
2. **Low-frequency synchronisation converges.** 26 rounds at H=10, loss
   3.92 → 0.18 across four processes with disjoint data shards.
3. **A cheap verifier catches naive poisoning.** Norm gate kills a 26x
   magnitude attack; direction gate kills a matched-magnitude random attack
   the norm gate cannot see.
4. **Churn is survivable.** SIGTERM a worker mid-run; the coordinator drops it
   and finishes with the remainder.
5. **int8 compression is nearly free.** 3.8x smaller payload, max absolute
   error 0.015 on the test model.

## 2. What it does not prove

- Nothing about **throughput at scale**. Measured on 2 CPU cores.
- Nothing about **GPU behaviour**: thermal throttling, WDDM, driver resets,
  PCIe pressure, mixed architectures.
- Nothing about **adversaries who are patient**. Both gates are defeated by an
  attacker that aligns with consensus and perturbs within tolerance.
- Nothing about **WAN reality**: no packet loss, no NAT, no relay, no jitter.
- Nothing about **model generality**: one tiny transformer, one data format.

Treat it as a design that compiles, not a system that ships.

---

## 3. The gap, subsystem by subsystem

Effort is in engineer-weeks for one competent systems engineer.

| # | Subsystem | Prototype | Production needs | Effort |
|---|---|---|---|---|
| 1 | Transport | Plain TCP, no auth, no retry | QUIC with UDP hole-punching (~90% direct success behind NAT), relay fallback, connection migration, multiplexed streams, checksums | 8–10 |
| 2 | Identity & auth | None — anyone who reaches the port registers | Ed25519 node identity, enrollment tokens, revocation, per-peer rate limits, signed updates | 4 |
| 3 | Scheduler | Static plan computed once | Dynamic rebalancing as nodes join/leave, admission control, priority queues, preemption, backfill | 10–12 |
| 4 | Verification | Norm gate + direction gate | Recompute sampling with tolerance, loss-impact scoring on held-out shards, reputation, stake/slash, measured false-positive rate | 16–20 |
| 5 | Client | CLI process | Windows/macOS/Linux installers, background service, sandboxed worker, thermal governor, sleep/wake, TDR recovery, bandwidth budget, auto-update, one-click uninstall | 24–30 |
| 6 | Coordinator | One in-process asyncio server | Postgres-backed job registry, HA (2+ replicas), multi-tenancy, quotas, billing events, audit log | 12–16 |
| 7 | Checkpointing | None — global state lives in RAM | Versioned object store, live peer-to-peer recovery, resharding on membership change, integrity verification | 8 |
| 8 | Pipeline parallelism | Not built | Stage placement by VRAM, activation compression, micro-batch pipelining, bubble minimisation | 16–20 |
| 9 | Observability | print() | Prometheus metrics, per-node dashboards, alerting, per-job cost attribution | 6 |
| 10 | Model & data layer | One toy transformer | HuggingFace transformers, tokenizer handling, streaming datasets, dedup, deterministic sharding, eval harness | 10 |
| 11 | Packaging & supply chain | pyproject.toml | Signed binaries, SBOM, reproducible builds, release channels, crash reporting | 8 |
| 12 | Docs & support tooling | README | Reference deployment, troubleshooting matrix, hardware compatibility list, diagnostic bundle | 6 |
| 13 | Telemetry ingest | SQLite + token header, no retention policy | Time-series store (Prometheus/ClickHouse), sampling, retention windows, backfill, schema versioning | 8 |
| 14 | Device identity | Random UUID, self-declared | Signed attestation of the hardware profile so a node cannot claim 16 GB it does not have; TPM/Secure Enclave where available | 10 |
| 15 | Dashboard | Single-tenant, read-only, one page | Multi-tenant views, auth/SSO, alerting on fleet health, per-job drill-down, export, mobile | 10 |
| 16 | Privacy & compliance | Opt-in location, coarse, tested | DPDP Act consent records, retention schedule, deletion API, DPA template, data-flow map for enterprise security review | 6 |
| 17 | GeoIP | Static city table | Licensed IP→city database at the edge, never storing the IP, with a documented accuracy radius | 3 |

**Total: 165–193 engineer-weeks, roughly 3.5 engineer-years.** With five
engineers that is 8–10 months to a product someone can pay for — which matches
the M2/M3 split in the design document.

## 4. The five things that will actually sink you

Ranked by probability of killing the project, not by how interesting they are.

**1. Verification economics.** Every scheme costs compute, and the cheap ones
are defeatable. If recompute sampling needs 1-in-8 redundancy, you have
permanently spent 12.5% of the network's capacity on trust. If you cannot get
below ~15%, the whole thing loses to renting A100s. Build the verifier before
you build the marketplace.

**2. Windows client support load.** Forty hardware configurations, each with
its own thermal and driver behaviour. One engineer permanently on support is
the realistic steady state. This is boring, expensive, and the single most
likely reason a competitor fails — which is also why it is the moat.

**3. Node supply.** A volunteer network with 100 nodes is a demo; with 10,000
it is infrastructure. Getting from one to the other requires either payment or
a reason people care. Neither is a technical problem and both take longer than
the engineering.

**4. Model diversity.** Your memory model assumes a dense transformer with
known parameter counts. MoE, multimodal, long-context and quantised variants
each break the planner's assumptions. Budget for a compatibility matrix that
grows every quarter.

**5. Trusting your own dashboard.** The fleet view is only as good as the
telemetry, and telemetry is self-reported. A node can claim 16 GB of VRAM it
does not have, and a customer making a capacity decision on that number will
notice when the run fails. Signed device attestation is the fix, and it is
worth doing before the dashboard becomes a sales artefact.

**6. Enterprise procurement.** Banks and large IT firms will ask about data
residency, DPDP Act compliance, source code escrow, indemnity and liability
for damaged hardware. None of that is engineering and all of it gates revenue.

## 5. Non-engineering requirements

- **Security review and penetration test.** ₹8–15 lakh. Non-negotiable before
  any enterprise pilot: you are shipping software that runs code on other
  people's machines and accepts files from strangers.
- **Sandboxing.** Every training payload must run in gVisor, Firecracker or at
  minimum a hardened container with no network egress. `weights_only=True` in
  this prototype is necessary and nowhere near sufficient.
- **Legal: contributor terms.** Explicit consent for GPU use, electricity,
  thermal wear, and data egress. Liability for a cooked laptop must be
  disclaimed clearly and honestly.
- **Privacy: DPDP Act 2023.** Telemetry is personal data once it can be
  linked to a person, and location makes that link easier. You need a lawful
  basis, a consent record per install, a retention schedule, and a deletion
  path. The prototype's opt-in design is the right shape; it is not yet a
  compliance artefact.
- **Legal: enterprise data.** A DPA, plus a documented statement that no
  training data leaves the customer's network in the on-prem deployment. That
  claim is a selling point in India and must be architecturally true.
- **Tax.** If you pay contributors: TDS under 194-O/194J, GST on the service,
  and — if you ever issue tokens — the VDA regime at 30% flat plus 1% TDS.
  Get an opinion before you design the payout system, not after.
- **Hiring.** Two systems engineers (Rust/C++, networking), two ML
  infrastructure engineers (PyTorch, FSDP), one Windows client engineer, one
  designer, one developer advocate. The first four are hard to find in India
  and worth paying above market for.

## 6. Build versus buy

| Component | Verdict |
|---|---|
| Transport / hole punching | **Buy** — `iroh` (QUIC, ~90% direct connection success) or PCCL |
| Collectives across WAN | **Buy** — PCCL (MIT) |
| Sync optimiser | **Buy** — DiLoCo from `prime`, later SparseLoCo |
| Compression | **Buy** — int8, then chunk Top-k + 2-bit from SparseLoCo |
| Sandbox | **Buy** — gVisor or Firecracker |
| Auto-update | **Buy** — TUF or Sparkle |
| **Planner / scheduler** | **Build** — nobody has it, it is your product |
| **Verification** | **Build** — nobody has it, it is your moat |
| **Client** | **Build** — nobody will, and it is what users touch |
| **Coordinator** | **Build** — it is the commercial surface |

The rule: build the three things no funded lab will build because they are not
research, and buy everything that is.

## 7. Three definitions of done

**Tier 1 — useful to one team (≈3 months after this prototype).**
One organisation can fine-tune a 7B model with LoRA across eight of its own
heterogeneous machines, from one command, surviving restarts. No
multi-tenancy, no verification, no payments. This is sellable as a consulting
engagement and proves the scheduler against a real fleet.

**Tier 2 — useful as a product (≈9–12 months).**
Multi-tenant coordinator, Windows client, recompute verification with a
measured false-positive rate, billing, SLA, on-prem deployment package. Two
paying customers. This is the raise.

**Tier 3 — useful as a network (2–3 years).**
Permissionless participation, economic layer, 1,000+ nodes, pipeline
parallelism for models larger than any single node's memory. This is the
original vision and the last thing to attempt.

## 8. Sequence, with gates

```
prototype ──► Tier 1 ──► Tier 2 ──► Tier 3
   now        3 months   9-12 mo    2-3 yrs
```

Do not skip a tier. Each one validates the assumption the next depends on:
Tier 1 validates the scheduler, Tier 2 validates the verifier and the buyer,
Tier 3 validates the economics. The most common failure in this space is
building Tier 3 and discovering at the end that Tier 1 never worked.
