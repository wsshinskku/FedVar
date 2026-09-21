"""Algorithm 2: select model norms within mean +/- population SD; average models."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Selection:
    norms: tuple[float, ...]
    mean: float
    std: float
    selected: tuple[int, ...]


def _validate(states):
    if not states or not states[0]:
        raise ValueError("At least one nonempty client state is required")
    first = states[0]
    for state in states:
        if state.keys() != first.keys():
            raise ValueError("Client state keys must match")
        for key, value in state.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != first[key].shape
                or value.dtype != first[key].dtype
            ):
                raise ValueError(f"Client tensor shapes and dtypes must match: {key}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite client tensor: {key}")


def select_inliers(states, parameter_keys=None):
    """Use concatenated parameter L2 norms and population (ddof=0) deviation.

    The training runner supplies model.named_parameters() keys; generic callers
    default to all floating keys. Both interval endpoints are included.
    """
    _validate(states)
    keys = (
        list(parameter_keys)
        if parameter_keys is not None
        else [key for key, value in states[0].items() if value.is_floating_point()]
    )
    if (
        not keys
        or len(set(keys)) != len(keys)
        or any(key not in states[0] or not states[0][key].is_floating_point() for key in keys)
    ):
        raise ValueError("Distinct floating parameter keys are required")
    norms = np.array(
        [
            torch.linalg.vector_norm(
                torch.cat([state[key].detach().cpu().double().reshape(-1) for key in keys])
            ).item()
            for state in states
        ],
        dtype=np.float64,
    )
    if not np.isfinite(norms).all():
        raise ValueError("Parameter norms overflowed; rescale client parameters")
    scale = max(float(norms.max()), np.finfo(float).tiny)
    mean = float((norms / scale).mean() * scale)
    std = float((norms / scale).std(ddof=0) * scale)
    if not np.isfinite([mean, std]).all():
        raise ValueError("Norm statistics overflowed")
    tolerance = 8 * np.finfo(float).eps * max(np.finfo(float).tiny, abs(mean), std)
    selected = tuple(
        np.flatnonzero(
            (norms >= mean - std - tolerance) & (norms <= mean + std + tolerance)
        ).tolist()
    )
    if not selected:
        raise ArithmeticError("No client lies in the computed standard-deviation interval")
    return Selection(tuple(norms.tolist()), mean, std, selected)


def aggregate(states, method="fedvar", sample_counts=None, parameter_keys=None):
    """Return a new unaliased state and aggregation diagnostics.

    FedVar uniformly averages selected clients, SDA(w)=sum(w)/n. Baselines
    sample-weight all clients. Floating buffers follow the same average;
    integer/bool buffers are copied from the first retained client.
    """
    _validate(states)
    if method not in ("fedvar", "fedavg", "fedprox", "fedsgd"):
        raise ValueError("Unknown aggregation method")
    if method == "fedvar":
        selection = select_inliers(states, parameter_keys)
        indices = selection.selected
        weights = np.full(len(indices), 1 / len(indices))
        diagnostics = {
            "norms": list(selection.norms),
            "mean": selection.mean,
            "population_std": selection.std,
            "selected": list(indices),
        }
    else:
        counts = (
            np.ones(len(states))
            if sample_counts is None
            else np.asarray(sample_counts, dtype=float)
        )
        if counts.shape != (len(states),) or not np.isfinite(counts).all() or np.any(counts <= 0):
            raise ValueError(
                "One finite positive sample count per participating client is required"
            )
        indices = tuple(range(len(states)))
        weights = counts / counts.max()
        weights = weights / weights.sum()
        diagnostics = {"selected": list(indices)}
    result = {}
    for key, first in states[0].items():
        if first.is_floating_point():
            value = torch.zeros_like(first, device="cpu", dtype=torch.float64)
            for index, weight in zip(indices, weights, strict=True):
                value.add_(states[index][key].detach().cpu().double(), alpha=float(weight))
            result[key] = value.to(first.dtype)
        else:
            result[key] = states[indices[0]][key].detach().cpu().clone()
    diagnostics["weights"] = weights.tolist()
    return result, diagnostics
