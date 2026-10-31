# Go to market

How to take mesh from "prototype someone could run" to "thing someone pays for".
This is written for a solo technical founder, not a company with a sales team.

---

## 1. What you are actually selling

Not "distributed training". That phrase has no buyer — the people who need it
don't know it exists, and the people who know it exists are building it
themselves.

You are selling **an answer to a budget question**: *can the GPUs we already
own train the model we want, and if not, what can they train?* That question
is asked by a person with a number to defend, and today they answer it by
guessing.

The wedge is the planner and the census. The training runtime is proof that
you understand the problem, not the thing you sell first.

## 2. Who buys, and what triggers the purchase

| Segment | The trigger | What they say | What they pay for |
|---|---|---|---|
| **University / research lab** | A grant for GPUs arrived, or a student needs to fine-tune a 7B on lab machines | "we have 12 machines, can we train on them?" | a capacity audit, then support |
| **Enterprise with idle workstations** | A cloud bill review, or a data-residency rule blocking offshore training | "we can't send this data out" | on-prem deployment + support contract |
| **Small AI team** | About to rent A100s | "is it cheaper to use what we have?" | the tool, self-serve |

The first two are where money actually is. The third is where GitHub stars
come from, which is what makes the first two take the meeting.

## 3. Positioning

> **mesh tells you what your fleet of consumer GPUs can actually train —
> before you spend a month finding out.**

Three sentences to have ready:

- *"Pooled VRAM is a fiction. A model has to fit on one machine, so your
  ceiling is your biggest GPU, not the sum of them."*
- *"mesh scans the machines and tells you the ceiling, the throughput and the
  ETA in about four seconds."*
- *"On a hundred consumer GPUs, eight can hold a 7B model. That number changes
  what people buy."*

Never lead with DiLoCo, gradient compression or P2P. Those are how it works,
not why anyone cares.

## 4. Pricing

Start with services, not software. A solo developer cannot support a SaaS
fleet, and a paid audit is revenue this month.

| Offer | Price | What it is |
|---|---|---|
| **Open source** | free | `mesh scan`, `mesh plan`, self-hosted dashboard up to 5 nodes. Apache-2.0 |
| **Capacity audit** | ₹75,000 / $900, one-off | You scan their fleet, deliver a 6-page report: ceiling, best model, projected ETA, recommended config, and what to buy if anything |
| **Team licence** | ₹25,000 / $300 per month | Hosted coordinator, unlimited nodes, 90-day retention, priority issues |
| **Enterprise / on-prem** | from ₹8L / $10k per year | Air-gapped install, SSO, support SLA, DPA, audit report for their security team |

The audit is the product-market-fit test. If nobody will pay ₹75k for the
answer, they will not pay for the software that produces it.

## 5. The ten-minute demo

Do not open with architecture. Open with their hardware.

1. **`mesh scan`** on the machine in front of them. Thirty seconds. *"This is
   your ceiling: 0.4B full, 5.3B with QLoRA. That's what this machine can do."*
2. **`mesh scan --fleet 100`.** *"Now here's your fleet. Eight of a hundred can
   hold Mistral-7B. Ninety-two cannot, and buying more of them won't help."*
3. **`mesh plan --params 7.2 --mode lora`.** *"And here's what it costs you in
   time: throughput, sync interval, ETA. This is the number that goes in the
   budget."*
4. **`mesh up --demo`.** The dashboard. *"This is every machine as the fleet
   sees it, with location only if the user opted in."*
5. Close with one question: **"Which of your machines should we scan first?"**

Step 5 is the whole sale. It converts a demo into a data-collection exercise
with their hardware, and the report that comes out of it is the paid audit.

## 6. Launch checklist

- [ ] Publish to PyPI as `mesh-train` — `pipx install mesh-train` must work
- [ ] Push the repo public with the README and the screenshot
- [ ] Host `site/index.html` on GitHub Pages, point a domain at it
- [ ] Record a 60-second screen capture of `mesh scan --fleet 100`
- [ ] Post to r/LocalLLaMA first — that audience owns the hardware in question
- [ ] Then Show HN, then LinkedIn, then X

**The post that will work:** *"I scanned 100 consumer GPUs. Only 8 of them can
run a 7B model."* Numbers in the title, a claim people will argue with, and the
tool is the way to check it. Do not write "introducing mesh, a framework for
distributed training".

The most useful thing a stranger can do is run `mesh scan` and paste the
output. Ask for exactly that in the post.

## 7. What to build next, decided by what sells

Do not build pipeline parallelism because it is interesting. Build what the
first three paying customers ask for. Expected order:

1. **A real model path.** Fine-tune a HuggingFace model with LoRA, not the toy
   transformer. This is the difference between a demo and a tool.
2. **Signed node identity.** Enterprises will ask who a node is before they
   ask how fast it is.
3. **A PDF report export.** The audit deliverable should generate itself.
4. **Pipeline parallelism.** Only when a customer has a model that does not
   fit and money to spend.

## 8. Honest risks

- **The free tier is very good.** Everything a buyer needs to make a decision
  works offline and free. That is deliberate — it is also why the audit, not
  the software, is the first product.
- **The market may be small.** Teams with heterogeneous idle GPU fleets are
  real but not numerous in India. Validate before building more.
- **A funded lab could ship this.** Prime Intellect, Nous and Macrocosmos all
  have the algorithms. None of them have a planner or a Windows client, and
  none of them are selling to Indian enterprises. That gap is the business.
