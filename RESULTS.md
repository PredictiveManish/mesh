# Measured results

Environment: 2 vCPU, 2 GB RAM, no GPU, PyTorch 2.15.0+cpu.
Model: 108,288-parameter transformer, 40-token vocab, seq 48, microbatch 4.
Every "worker" below is a separate OS process with its own data shard.

## 1. Convergence, 4 workers

`mesh run --workers 4 --rounds 26 --H 10 --lr 3e-3`

```
round   0  loss 3.9196  (start)
round   5  loss 3.2234  updates 4/4  rej 0    4.9s
round  10  loss 2.9220  updates 4/4  rej 0    9.7s
round  15  loss 1.4430  updates 4/4  rej 0   14.6s
round  20  loss 0.5693  updates 4/4  rej 0   19.7s
round  26  loss 0.1833  updates 4/4  rej 0   25.5s
wall 25.5 s
```

Cross-entropy on a 40-way vocab starts at ln(40)=3.69. The model reaches 0.18,
i.e. it learned the synthetic language. DiLoCo with H=10 (sync every 10 local
steps) does not prevent convergence.

## 2. Byzantine worker, large-noise attack

`mesh run --workers 4 --rounds 12 --byzantine 1 --byzantine-kind noise`

```
[verify] REJECT w0: norm 26.2x the median (gate 3.0x)   x12 rounds
loss 3.9196 -> 2.9528   rejected 12/12
```

Without the gate, one worker with a 26x-norm update among four moves the
global model by roughly 6x the honest step every round. The run would diverge.

## 3. Byzantine worker, matched-magnitude attack

`mesh run --workers 4 --rounds 10 --byzantine 1 --byzantine-kind subtle`

```
[verify] REJECT w0: direction off consensus (cos +0.00)   x10 rounds
loss 3.9196 -> 3.0927   rejected 10/10
```

The attacker normalises its random vector to the honest update's norm, so the
norm gate passes it. The direction gate catches it: cosine ~0.00 against the
consensus, where honest workers sit near +0.99.

## 4. Churn

`mesh run --workers 4 --rounds 20 --churn-after 6`

```
[churn] killed w0 after round 6
round  10  loss 3.0592  updates 3/3
round  20  loss 0.6522  updates 3/3
dropped mid-run: w0
```

