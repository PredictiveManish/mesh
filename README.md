<div align="center">

# mesh

**Know what your GPUs can actually train — before you spend a month finding out.**

Scan a fleet of consumer GPUs, get the real ceiling, the throughput and the ETA.
Then train across them over the internet, when it fits.

`python` · `PyTorch` · `Apache-2.0` · `v0.1.0`

</div>

---

## The problem nobody prices in

Every "we have 600 GB of pooled VRAM" spreadsheet is fiction. A model has to
fit on **one** node before it can be trained on any of them, so the ceiling is
set by your largest single GPU — not by the sum.

On a 100-machine fleet of 4 / 6 / 8 / 16 GB consumer cards, 646 GB pooled:

```
  model                 params    full   QLoRA  nodes that fit
  ------------------------------------------------------------
  Qwen2.5-0.5B            0.5B     yes     yes       100 / 100
  Llama-3.2-1B            1.2B       —     yes       100 / 100
  Qwen2.5-3B              3.0B       —     yes        60 / 100
  Mistral-7B              7.2B       —     yes         8 / 100
  Llama-3.1-8B            8.0B       —     yes         8 / 100

  Largest full fine-tune that fits:  Qwen2.5-0.5B
  Largest QLoRA fine-tune that fits: Llama-3.1-8B
```

Eight machines out of a hundred can hold Mistral-7B. The other ninety-two
cannot, and adding more of them will never change that.

Teams discover this after a week of setup and a month of failed runs. `mesh`
answers it in four seconds, from the machines themselves.

## Three commands

```bash
mesh scan                      # what can this machine train?
mesh scan --fleet 100          # ...or a whole fleet
mesh plan --params 7.2 --mode lora --tokens 2   # throughput, sync interval, ETA
mesh up --demo                 # coordinator + live dashboard
```

## Install

```bash
pipx install mesh-train      # or: pip install mesh-train
mesh up --demo               # dashboard at http://localhost:8080
```

Or with Docker:

```bash
docker compose up --build
```

Adding real machines to a fleet:

```bash
mesh agent --server http://your-host:8080 --location Bengaluru --consent-location
```

The agent prints exactly what it will share before it shares it, then
heartbeats every 30 seconds.

## The dashboard

`mesh up` serves a live fleet view: online now, active 24 h, pooled VRAM, a
capacity ladder showing where the fleet falls off, new installs per day, the
VRAM tier mix, GPU inventory, and city-level geography (opt-in, coarse).

`dashboard/index.html` also opens as a plain file with a snapshot baked in.

## What it answers that nothing else does

| Question | mesh | Typical answer today |
|---|---|---|
| What is the largest model this fleet can hold? | computed per node from the census | guess, then find out |
| How many of my machines can run a 7B? | counted exactly | "we have 646 GB pooled" |
| How often should nodes synchronise? | auto-tuned so upload ≤ 20% of compute | fixed constant from a paper |
| How long until the run finishes? | ETA from measured throughput | unknown |

## Who it is for

- **Labs and universities** with a mixed bag of GPUs and no cluster team.
- **Enterprises** with idle workstations and a data-residency rule that rules
  out offshore cloud.
- **Anyone** deciding whether to rent an A100 or use what they already own.

It is **not** for training frontier models. A hundred mixed laptops are worth
roughly one H100 of effective throughput. That is a real number, and `mesh`
will tell you so.

## Status: what is real

| Component | State |
|---|---|
| Capacity planner (heterogeneous fleet → placement, throughput, ETA) | **works** |
| `mesh scan` / `plan` / `up` / `agent` | **works** |
| Fleet registry, HTTP API, dashboard | **works** |
| DiLoCo training loop, int8 compression, churn-tolerant barrier | **works** (reference implementation, small models) |
| Robust aggregation (magnitude + consensus-direction gates) | **works** — a gate, not a proof |
| Pipeline parallelism, signed node identity, WAN transport | **not built** |

The training runtime is honest about being a reference implementation. The
planner and the census are the parts you can rely on today.

## How it works

```
 census ──► planner ──► placement + H + ETA
   │           │
   │           └─► run: DiLoCo outer loop, int8 pseudo-gradients
   │                       │
   └─► registry ──► dashboard
```

- **Census** probes the device: VRAM, measured throughput, uplink, RAM, disk.
- **Planner** computes what fits, how fast, and how often to sync.
- **Runtime** trains with DiLoCo — local steps, rare synchronisation — so
  ordinary internet links are enough.
- **Registry** aggregates the fleet for the dashboard and for planning.

## Map data

The dashboard silhouette follows the official Indian depiction of the country
(includes Jammu & Kashmir, Ladakh, Aksai Chin and the island groups), derived
from the DataMeet India composite and simplified for display. It is not a
substitute for Survey of India map products.

## Privacy

Node IDs are random UUIDs, not hardware fingerprints. No email, hostname, MAC
or IP is stored. Location is opt-in, city-level and rounded to about 1 km; the
fields are absent entirely unless the user consents. Enforced in code, with
tests.

## Documentation

- [`docs/PRODUCTION-READINESS.md`](docs/PRODUCTION-READINESS.md) — the honest gap between this and a product
- [`docs/GO-TO-MARKET.md`](docs/GO-TO-MARKET.md) — who buys this, and for how much
- [`RESULTS.md`](RESULTS.md) — measured numbers from real runs
- [`CHANGELOG.md`](CHANGELOG.md)

## Contributing

The most useful contribution right now is a `mesh scan` result from unusual
hardware. Open an issue with the output and your GPU model.

## Licence

Apache-2.0. The client and library stay permissive on purpose — the agent runs
on other people's machines and has to be auditable.
