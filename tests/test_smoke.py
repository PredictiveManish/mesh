"""Fast checks on the pieces that are easy to get subtly wrong."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mesh.census import synthetic_fleet
from mesh.planner import plan, max_params_on
from mesh.protocol import compress, decompress, OuterOptimizer
from mesh.types import JobSpec
from mesh.verify import RobustAggregator
from mesh import model as M


def test_memory_floor_is_honest():
    fleet = synthetic_fleet(20)
    small = min(fleet, key=lambda n: n.vram_gb)
    big = max(fleet, key=lambda n: n.vram_gb)
    # 16 bytes/param means a 16 GB card holds ~0.8B params, not 4B
    assert 0.7e9 < max_params_on(big, JobSpec()) < 0.9e9
    assert max_params_on(small, JobSpec()) < 0.25e9


def test_planner_excludes_nodes_that_cannot_hold_the_model():
    fleet = synthetic_fleet(40)
    job = JobSpec(params=0.5e9, target_tokens=1e9, compression="int8")
    p = plan(job, fleet)
    assert p.replica_count > 0
    assert len(p.excluded) > 0
    for rid in p.replicas:
        node = next(n for n in fleet if n.node_id == rid[0])
        assert max_params_on(node, job) >= job.params


def test_lora_fits_more_than_full():
    fleet = synthetic_fleet(40)
    full = plan(JobSpec(params=8e9, target_tokens=1e9, train_mode="full"), fleet)
    lora = plan(JobSpec(params=8e9, target_tokens=1e9, train_mode="lora"), fleet)
    assert full.replica_count == 0
    assert lora.replica_count > 0


def test_int8_roundtrip_is_close():
    sd = M.build().state_dict()
    delta = {k: torch.randn_like(v) for k, v in sd.items()}
    back = decompress(compress(delta, "int8"))
    err = max(float((back[k] - delta[k]).abs().max()) for k in delta)
    assert err < 0.05, err


def test_outer_optimizer_moves_toward_local_model():
    sd = M.build().state_dict()
    local = {k: v + 0.01 for k, v in sd.items()}      # moved downhill
    delta = {k: sd[k] - local[k] for k in sd}
    opt = OuterOptimizer(sd, lr=1.0, beta=0.0)
    new = opt.step(sd, delta)
    for k in sd:
        assert torch.allclose(new[k], local[k], atol=1e-5)


def test_verifier_rejects_noise_and_keeps_honest():
    sd = M.build().state_dict()
    honest = {k: torch.randn_like(v) for k, v in sd.items()}
    updates = {
        "a": honest,
        "b": {k: v * 1.05 for k, v in honest.items()},
        "c": {k: v * 0.95 for k, v in honest.items()},
        "evil": {k: torch.randn_like(v) * 50 for k, v in sd.items()},
    }
    _, verdicts = RobustAggregator().aggregate(updates)
    by_id = {v.node_id: v for v in verdicts}
    assert by_id["evil"].accepted is False
    assert by_id["a"].accepted and by_id["b"].accepted and by_id["c"].accepted


def test_corpus_has_learnable_structure():
    c = M.make_corpus(2000)
    assert c.min() >= 0 and c.max() < M.VOCAB
    assert len(torch.unique(c)) > 20
