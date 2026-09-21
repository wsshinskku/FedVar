import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    algorithm: str = "fedvar"
    dataset: str = "synthetic"
    model: str = "cnn"
    seed: int = 7
    clients: int = 8
    clients_per_round: int = 4
    rounds: int = 3
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.03
    momentum: float = 0.0
    weight_decay: float = 0.0
    proximal_mu: float = 0.01
    partition: str = "dirichlet"
    dirichlet_alpha: float = 0.5
    shards_per_client: int = 2
    evaluate_every: int = 1
    device: str = "cpu"
    threads: int = 2
    data_dir: str = "data"
    download: bool = False
    synthetic_train: int = 512
    synthetic_test: int = 128
    train_limit: int | None = None
    test_limit: int | None = None

    def validate(self):
        if self.algorithm not in ("fedvar", "fedavg", "fedsgd", "fedprox"):
            raise ValueError("Unknown algorithm")
        if self.dataset not in ("synthetic", "mnist", "cifar10", "cifar100"):
            raise ValueError("Unknown dataset")
        if self.model not in ("cnn", "mlp", "tinynet_a", "ghostnet_100", "mobilenetv3_small_100"):
            raise ValueError("Unknown model")
        if self.partition not in ("iid", "dirichlet", "shards"):
            raise ValueError("Unknown partition")
        for name in (
            "clients",
            "clients_per_round",
            "rounds",
            "local_epochs",
            "batch_size",
            "shards_per_client",
            "evaluate_every",
            "threads",
            "synthetic_train",
            "synthetic_test",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.clients_per_round > self.clients:
            raise ValueError("clients_per_round exceeds clients")
        for name in ("learning_rate", "dirichlet_alpha"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("momentum", "weight_decay", "proximal_mu"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.momentum >= 1:
            raise ValueError("momentum must be below one")
        for name in ("train_limit", "test_limit"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or null")
        if not isinstance(self.download, bool):
            raise ValueError("download must be boolean")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be cpu or cuda[:index]")
        return self

    def as_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path=None, **overrides):
        values = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values).validate()
