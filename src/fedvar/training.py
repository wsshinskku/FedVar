"""Synchronous federation with isolated clients and post-aggregation server evaluation."""

import copy
import csv
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .aggregation import aggregate
from .config import Config
from .data import load_datasets, partition_labels
from .models import build_model


def _forward_local(model, images):
    """Keep singleton examples; BN uses stored statistics for those batches."""
    changed = []
    if len(images) == 1:
        for module in model.modules():
            if (
                isinstance(
                    module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
                )
                and module.training
            ):
                if not module.track_running_stats:
                    raise ValueError("Singleton local batches require BatchNorm running statistics")
                changed.append(module)
                module.eval()
    try:
        return model(images)
    finally:
        for module in changed:
            module.train()


def local_update(global_model, dataset, config, seed):
    if len(dataset) == 0:
        raise ValueError("Local training dataset must not be empty")
    model = copy.deepcopy(global_model).to(config.device)
    model.train()
    anchor = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    examples, total_loss = 0, 0.0
    if config.algorithm == "fedsgd":
        # One full-data gradient step; sample-weighting states then equals one
        # global gradient step for models without minibatch-dependent buffers.
        optimizer.zero_grad(set_to_none=True)
        for images, labels in loader:
            images, labels = images.to(config.device), labels.to(config.device)
            loss = nn.functional.cross_entropy(
                _forward_local(model, images), labels, reduction="sum"
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Local objective became nonfinite")
            (loss / len(dataset)).backward()
            total_loss += float(loss.detach())
            examples += len(labels)
        optimizer.step()
    else:
        for _ in range(config.local_epochs):
            for images, labels in loader:
                images, labels = images.to(config.device), labels.to(config.device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(_forward_local(model, images), labels)
                if config.algorithm == "fedprox":
                    loss = loss + config.proximal_mu / 2 * sum(
                        (p - a).square().sum()
                        for p, a in zip(model.parameters(), anchor, strict=True)
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Local objective became nonfinite")
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(labels)
                examples += len(labels)
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }, total_loss / examples


def evaluate_model(model, dataset, device="cpu", batch_size=256):
    if len(dataset) == 0:
        raise ValueError("Evaluation dataset must not be empty")
    model = model.to(device)
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss_sum += float(nn.functional.cross_entropy(output, labels, reduction="sum"))
            correct += int((output.argmax(1) == labels).sum())
            total += len(labels)
    return {"test_loss": loss_sum / total, "test_accuracy": correct / total, "test_examples": total}


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train(config, output):
    config.validate()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.use_deterministic_algorithms(True)
    train_set, test_set, labels, classes, channels, size = load_datasets(config)
    partitions = partition_labels(
        labels,
        config.clients,
        config.partition,
        config.dirichlet_alpha,
        config.shards_per_client,
        config.seed + 201,
    )
    model = build_model(config.model, channels, classes, size).to(config.device)
    keys = tuple(dict(model.named_parameters()))
    rng = np.random.default_rng(config.seed + 202)
    history = [{"round": 0, **evaluate_model(model, test_set, config.device)}]
    _write(output / "config.json", config.as_dict())
    _write(
        output / "partition.json",
        {
            "indices": partitions,
            "class_counts": [
                np.bincount(labels[part], minlength=classes).tolist() for part in partitions
            ],
        },
    )
    for round_index in range(1, config.rounds + 1):
        participants = sorted(
            rng.choice(config.clients, config.clients_per_round, replace=False).tolist()
        )
        states, losses = [], []
        for client in participants:
            state, loss = local_update(
                model,
                Subset(train_set, partitions[client]),
                config,
                config.seed + round_index * 100003 + client,
            )
            states.append(state)
            losses.append(loss)
        combined, diagnostics = aggregate(
            states, config.algorithm, [len(partitions[client]) for client in participants], keys
        )
        model.load_state_dict(combined)
        row = {
            "round": round_index,
            "local_loss": float(np.mean(losses)),
            "participants": participants,
            "retained_clients": [participants[i] for i in diagnostics["selected"]],
            "aggregation": diagnostics,
        }
        if round_index % config.evaluate_every == 0 or round_index == config.rounds:
            row.update(evaluate_model(model, test_set, config.device))
        history.append(row)
        accuracy = f" accuracy={row['test_accuracy']:.4f}" if "test_accuracy" in row else ""
        print(
            f"Round {round_index}/{config.rounds}: retained={len(diagnostics['selected'])}/{len(participants)}{accuracy}",
            flush=True,
        )
    summary = {
        "algorithm": config.algorithm,
        "config": config.as_dict(),
        "train_examples": len(train_set),
        "test_examples": len(test_set),
        "history": history,
        "final": history[-1],
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
        },
    }
    _write(output / "metrics.json", summary)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["round", "local_loss", "test_loss", "test_accuracy"]
        )
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in writer.fieldnames} for row in history)
    torch.save(
        {
            "format_version": 1,
            "config": config.as_dict(),
            "round": config.rounds,
            "state_dict": {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            },
            "classes": classes,
            "channels": channels,
            "image_size": size,
        },
        output / "checkpoint.pt",
    )
    return summary


def evaluate_checkpoint(checkpoint, device="cpu"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    config = Config(**saved["config"]).validate()
    config.device = device
    config.validate()
    torch.set_num_threads(config.threads)
    _, test_set, _, classes, channels, size = load_datasets(config)
    model = build_model(config.model, channels, classes, size)
    model.load_state_dict(saved["state_dict"])
    return {"round": saved["round"], **evaluate_model(model, test_set, device)}
