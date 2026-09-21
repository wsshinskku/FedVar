"""Disjoint benchmark test sets and explicit, reproducible client partitions."""

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset


def partition_labels(labels, clients, mode="dirichlet", alpha=0.5, shards_per_client=2, seed=0):
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) < clients or clients < 1:
        raise ValueError("At least one training example per client is required")
    rng = np.random.default_rng(seed)
    if mode == "iid":
        groups = [part.tolist() for part in np.array_split(rng.permutation(len(labels)), clients)]
    elif mode == "shards":
        if clients * shards_per_client > len(labels) or shards_per_client < 1:
            raise ValueError("Not enough examples for the requested nonempty label shards")
        order = rng.permutation(len(labels))
        sorted_indices = order[np.argsort(labels[order], kind="stable")]
        shards = np.array_split(sorted_indices, clients * shards_per_client)
        assignment = rng.permutation(len(shards)).reshape(clients, shards_per_client)
        groups = [np.concatenate([shards[index] for index in row]).tolist() for row in assignment]
    elif mode == "dirichlet":
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Dirichlet alpha must be positive and finite")
        groups = [[] for _ in range(clients)]
        for label in np.unique(labels):
            indices = rng.permutation(np.flatnonzero(labels == label))
            counts = rng.multinomial(len(indices), rng.dirichlet(np.full(clients, alpha)))
            for client, part in enumerate(np.split(indices, np.cumsum(counts)[:-1])):
                groups[client].extend(part.tolist())
        for group in groups:
            if not group:
                donor = max(groups, key=len)
                group.append(donor.pop())
    else:
        raise ValueError("Unknown partition mode")
    for group in groups:
        rng.shuffle(group)
    return groups


def load_datasets(config):
    if config.dataset == "synthetic":
        classes, channels, size = 10, 1, 28
        generator = torch.Generator().manual_seed(config.seed + 101)
        prototypes = torch.randn(classes, channels, size, size, generator=generator)

        def make(count, seed):
            local = torch.Generator().manual_seed(seed)
            labels = torch.arange(count) % classes
            labels = labels[torch.randperm(count, generator=local)]
            images = prototypes[labels] + 0.3 * torch.randn(
                count, channels, size, size, generator=local
            )
            return TensorDataset(images, labels), labels.numpy()

        train, targets = make(config.synthetic_train, config.seed + 102)
        test, _ = make(config.synthetic_test, config.seed + 103)
    else:
        try:
            from torchvision import datasets, transforms
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "Install matching PyTorch/torchvision builds with pip install -e '.[vision]'"
            ) from exc
        channels, size = (1, 28) if config.dataset == "mnist" else (3, 32)
        classes = 100 if config.dataset == "cifar100" else 10
        dataset_type = {
            "mnist": datasets.MNIST,
            "cifar10": datasets.CIFAR10,
            "cifar100": datasets.CIFAR100,
        }[config.dataset]
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5] * channels, [0.5] * channels)]
        )
        train = dataset_type(
            config.data_dir, train=True, transform=transform, download=config.download
        )
        test = dataset_type(
            config.data_dir, train=False, transform=transform, download=config.download
        )
        targets = np.asarray(train.targets)
    if config.train_limit is not None and config.train_limit < len(train):
        indices = np.random.default_rng(config.seed + 104).choice(
            len(train), config.train_limit, replace=False
        )
        train, targets = Subset(train, indices.tolist()), targets[indices]
    if config.test_limit is not None and config.test_limit < len(test):
        indices = np.random.default_rng(config.seed + 105).choice(
            len(test), config.test_limit, replace=False
        )
        test = Subset(test, indices.tolist())
    return train, test, np.asarray(targets), classes, channels, size
