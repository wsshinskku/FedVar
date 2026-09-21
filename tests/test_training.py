import copy
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Subset, TensorDataset

from fedvar.aggregation import aggregate
from fedvar.cli import main
from fedvar.config import Config
from fedvar.data import load_datasets, partition_labels
from fedvar.training import evaluate_checkpoint, local_update, train


def small(**overrides):
    values = {
        "model": "mlp",
        "clients": 4,
        "clients_per_round": 4,
        "rounds": 2,
        "local_epochs": 1,
        "batch_size": 8,
        "synthetic_train": 40,
        "synthetic_test": 20,
        "threads": 1,
    }
    return Config(**{**values, **overrides}).validate()


@pytest.mark.parametrize("mode", ["iid", "dirichlet", "shards"])
def test_partitions_are_disjoint_exhaustive_nonempty_and_repeatable(mode):
    labels = np.arange(100) % 10
    first = partition_labels(labels, 10, mode, alpha=0.01, seed=9)
    assert first == partition_labels(labels, 10, mode, alpha=0.01, seed=9)
    assert all(first)
    assert sorted(index for group in first for index in group) == list(range(100))


def test_synthetic_test_samples_are_separate_and_reproducible():
    config = small()
    train_set, test_set, labels, *_ = load_datasets(config)
    repeated_train, repeated_test, repeated_labels, *_ = load_datasets(config)
    torch.testing.assert_close(train_set.tensors[0], repeated_train.tensors[0])
    torch.testing.assert_close(test_set.tensors[0], repeated_test.tensors[0])
    np.testing.assert_array_equal(labels, repeated_labels)
    assert not torch.equal(train_set.tensors[0][:20], test_set.tensors[0])


def test_client_update_keeps_global_model_unchanged():
    model = nn.Linear(2, 2)
    initial = copy.deepcopy(model.state_dict())
    dataset = TensorDataset(torch.tensor([[1.0, 2.0], [2.0, 1.0]]), torch.tensor([0, 1]))
    state, _ = local_update(model, dataset, small(), 3)
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, initial[key])
        assert tensor.data_ptr() != state[key].data_ptr()
    assert any(not torch.equal(state[key], initial[key]) for key in initial)


@pytest.mark.parametrize("algorithm", ["fedvar", "fedsgd"])
def test_singleton_batchnorm_client_preserves_example_and_affine_learning(algorithm):
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        model[1].weight.copy_(torch.eye(2))
    before = copy.deepcopy(model.state_dict())
    dataset = TensorDataset(torch.tensor([[1.0, 2.0]]), torch.tensor([0]))
    state, loss = local_update(model, dataset, small(algorithm=algorithm), 3)
    assert np.isfinite(loss)
    assert not torch.equal(state["0.weight"], before["0.weight"])
    assert not torch.equal(state["1.weight"], before["1.weight"])
    torch.testing.assert_close(state["0.running_mean"], before["0.running_mean"])
    assert state["0.num_batches_tracked"].item() == 0
    for key in before:
        torch.testing.assert_close(model.state_dict()[key], before[key])


def test_batchnorm_updates_regular_batches_and_retains_singleton_remainder():
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2))
    data = TensorDataset(
        torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]), torch.tensor([0, 1, 0])
    )
    state, _ = local_update(model, data, small(batch_size=2), 3)
    assert state["0.num_batches_tracked"].item() == 1


def test_fedsgd_equals_pooled_single_gradient_step_with_unequal_client_sizes():
    torch.manual_seed(8)
    model = nn.Linear(2, 2)
    direct = copy.deepcopy(model)
    dataset = TensorDataset(torch.randn(7, 2), torch.tensor([0, 0, 1, 1, 0, 1, 1]))
    config = small(algorithm="fedsgd", local_epochs=5, batch_size=2)
    clients = [Subset(dataset, [0, 1]), Subset(dataset, [2, 3, 4, 5, 6])]
    states = [local_update(model, data, config, seed=6)[0] for data in clients]
    combined, _ = aggregate(states, "fedsgd", [2, 5])
    optimizer = torch.optim.SGD(direct.parameters(), lr=config.learning_rate)
    optimizer.zero_grad()
    nn.functional.cross_entropy(direct(dataset.tensors[0]), dataset.tensors[1]).backward()
    optimizer.step()
    for key, tensor in direct.state_dict().items():
        torch.testing.assert_close(combined[key], tensor)


@pytest.mark.parametrize("algorithm", ["fedvar", "fedavg", "fedsgd", "fedprox"])
def test_training_updates_server_and_checkpoint_evaluates_same_model(tmp_path, algorithm):
    config = small(algorithm=algorithm)
    output = tmp_path / algorithm
    result = train(config, output)
    evaluation = evaluate_checkpoint(output / "checkpoint.pt")
    assert evaluation["test_accuracy"] == result["final"]["test_accuracy"]
    assert evaluation["test_loss"] == pytest.approx(result["final"]["test_loss"])
    assert result["history"][0]["test_loss"] != result["final"]["test_loss"]
    assert (output / "partition.json").exists()
    assert (output / "metrics.csv").exists()


def test_fixed_seed_repeats_participation_and_final_cpu_model(tmp_path):
    first = train(small(), tmp_path / "first")
    second = train(small(), tmp_path / "second")
    assert first["history"] == second["history"]
    with pytest.raises(FileExistsError):
        train(small(), tmp_path / "first")


def test_cli_comparison_keeps_config_seed_and_partition_fixed(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(small(rounds=1).as_dict()), encoding="utf-8")
    output = tmp_path / "comparison"
    main(
        [
            "compare",
            "--config",
            str(path),
            "--output",
            str(output),
            "--algorithms",
            "fedavg",
            "fedvar",
        ]
    )
    assert json.loads((output / "fedavg/partition.json").read_text()) == json.loads(
        (output / "fedvar/partition.json").read_text()
    )
    assert set(json.loads((output / "comparison.json").read_text())) == {"fedavg", "fedvar"}


@pytest.mark.parametrize(
    "changes",
    [
        {"clients_per_round": 5},
        {"learning_rate": 0},
        {"rounds": 0},
        {"seed": -1},
        {"dirichlet_alpha": float("nan")},
    ],
)
def test_config_rejects_invalid_experiments(changes):
    with pytest.raises(ValueError):
        small(**changes)
