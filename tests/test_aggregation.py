import copy

import numpy as np
import pytest
import torch

from fedvar.aggregation import aggregate, select_inliers


def states(values):
    return [{"weight": torch.tensor([value], dtype=torch.float64)} for value in values]


def test_paper_rule_discards_outlying_norm_then_averages_actual_tensors():
    local = states([1, 2, 3, 100])
    result, info = aggregate(local)
    assert info["selected"] == [0, 1, 2]
    torch.testing.assert_close(result["weight"], torch.tensor([2.0], dtype=torch.float64))
    np.testing.assert_allclose(info["norms"], [1, 2, 3, 100])
    assert info["population_std"] == pytest.approx(np.std([1, 2, 3, 100], ddof=0))
    assert info["weights"] == pytest.approx([1 / 3] * 3)


@pytest.mark.parametrize("values", [[0, 0, 0], [4, 4, 4], [1, 3], [5]])
def test_equal_zero_single_and_inclusive_boundary_cases_keep_clients(values):
    selection = select_inliers(states(values))
    assert selection.selected == tuple(range(len(values)))
    assert aggregate(states(values))[0]["weight"].item() == pytest.approx(np.mean(values))


def test_filter_is_relative_even_for_very_small_model_norms():
    assert select_inliers(states([1e-18, 2e-18, 3e-18, 100e-18])).selected == (0, 1, 2)


def test_norm_includes_all_parameters_and_does_not_use_signed_average():
    local = [
        {"a": torch.tensor([3.0]), "b": torch.tensor([-4.0])},
        {"a": torch.tensor([-3.0]), "b": torch.tensor([4.0])},
    ]
    selection = select_inliers(local)
    assert selection.norms == (5.0, 5.0)
    result, _ = aggregate(local)
    assert result["a"].item() == result["b"].item() == 0


def test_aggregation_never_modifies_or_aliases_client_inputs():
    local = states([1, 2, 3, 100])
    before = copy.deepcopy(local)
    result, _ = aggregate(local)
    for original, saved in zip(local, before, strict=True):
        torch.testing.assert_close(original["weight"], saved["weight"])
        assert result["weight"].data_ptr() != original["weight"].data_ptr()
    result["weight"].zero_()
    assert local[0]["weight"].item() == 1


def test_named_parameter_norms_exclude_buffers_with_explicit_buffer_rule():
    local = states([1, 2, 3, 100])
    for index, state in enumerate(local):
        state["running_mean"] = torch.tensor([1000.0 * (index + 1)])
        state["counter"] = torch.tensor(index + 1)
    combined, info = aggregate(local, parameter_keys=["weight"])
    assert info["selected"] == [0, 1, 2]
    assert combined["running_mean"].item() == 2000
    assert combined["counter"].dtype == torch.int64
    assert combined["counter"].item() == 1


@pytest.mark.parametrize("method", ["fedavg", "fedprox", "fedsgd"])
def test_baselines_weight_all_clients_by_sample_count(method):
    result, info = aggregate(states([1, 3]), method, sample_counts=[1, 3])
    assert result["weight"].item() == 2.5
    assert info["weights"] == [0.25, 0.75]
    assert info["selected"] == [0, 1]


def test_large_sample_counts_are_normalized_without_overflow():
    result, info = aggregate(states([1, 3]), "fedavg", sample_counts=[1e308, 1e308])
    assert result["weight"].item() == 2
    assert sum(info["weights"]) == 1


@pytest.mark.parametrize(
    "local",
    [
        [],
        [{"w": torch.tensor([float("nan")])}],
        [{"w": torch.ones(2)}, {"w": torch.ones(3)}],
        [{"w": torch.ones(2)}, {"other": torch.ones(2)}],
        [{"w": torch.ones(2)}, {"w": torch.ones(2, dtype=torch.float64)}],
    ],
)
def test_invalid_states_fail_clearly(local):
    with pytest.raises(ValueError):
        aggregate(local)


@pytest.mark.parametrize("counts", [[0, 1], [-1, 1], [1], [np.inf, 1]])
def test_invalid_baseline_weights_fail_clearly(counts):
    with pytest.raises(ValueError):
        aggregate(states([1, 3]), "fedavg", counts)
