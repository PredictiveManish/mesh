"""Core data types shared by every plane of the system."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class NodeProfile:
    """What a machine can actually do. Measured, not nominal."""

    node_id: str
    vram_gb: float = 0.0
    tflops_bf16: float = 0.0
    uplink_mbps: float = 0.0
    downlink_mbps: float = 0.0
    reliability: float = 0.90
    thermal_headroom: float = 0.80
    kind: str = "cpu"
    host: str = "localhost"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NodeProfile":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def summary(self) -> str:
        return (
            f"{self.node_id:<16} {self.kind:<4} "
            f"{self.vram_gb:5.1f} GB  {self.tflops_bf16:6.1f} TF  "
            f"{self.uplink_mbps:6.1f}/{self.downlink_mbps:6.1f} Mbps"
        )


@dataclass
class JobSpec:
    """What the user asked for."""

    model: str = "tiny-char-lm"
    method: str = "diloco"          # diloco | sparseloco | pipeline
    params: float = 120_000.0
    target_tokens: float = 0.0
    sync_every: int = 0             # 0 = let the planner choose
    compression: str = "int8"       # none | int8 | topk
    seq_len: int = 48
    microbatch: int = 0             # 0 = heuristic
    bytes_per_param: float = 16.0   # weights+grads+adam m,v, fp32
    train_mode: str = "full"        # full | lora
    adapter_fraction: float = 0.02  # share of params that are trainable in lora


@dataclass
class Placement:
    """The planner's answer."""

    replicas: list[list[str]] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    model_parallel: int = 1
    per_node_microbatch: dict[str, int] = field(default_factory=dict)
    sync_every: int = 1
    expected_tps: float = 0.0
    round_seconds: float = 0.0
    bottleneck: str = ""
    rounds: int = 0
    eta_hours: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def replica_count(self) -> int:
        return len(self.replicas)


@dataclass
class Verdict:
    """One worker's update, judged."""

    node_id: str
    accepted: bool
    reason: str = ""
    norm: float = 0.0
    norm_ratio: float = 0.0
    cosine: float = 0.0
