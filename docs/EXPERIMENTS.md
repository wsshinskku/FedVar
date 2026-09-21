# Experiments and reproducibility

## Published settings and runnable presets

Section III of the [paper](../FedVar__Federated_Learning_Algorithm_with_Weight_Variation_in_Clients.pdf) specifies 100 clients, 10 sampled clients per round, local batch size 50, 5 local epochs, 200 communication rounds, and evaluation every 20 rounds. It names MNIST, CIFAR-10, CIFAR-100 and the TinyNet, GhostNet, and MobileNetV3 model families through the experimental platform.

| Configuration | Purpose |
|---|---|
| `configs/smoke.json` | Offline synthetic images; fast end-to-end verification |
| `configs/mnist.json` | Practical MNIST experiment with 20 clients and 20 rounds |
| `configs/paper-budget-mnist.json` | Published round/client/batch/epoch budget, with explicit MNIST + CNN + Dirichlet-0.5 choices |

The MNIST presets select the included CNN, SGD with learning rate 0.01, seed 7, and a Dirichlet-0.5 client partition. All architecture, optimizer, seed, and partition choices are explicit in the saved configuration. Report experiment results together with these resolved settings and the generated client indices.

## Run

```bash
pip install -e '.[vision,dev]'
fedvar train --config configs/mnist.json --download --output runs/mnist-fedvar
fedvar compare --config configs/mnist.json --download --output runs/mnist-comparison
fedvar train --config configs/paper-budget-mnist.json --download --output runs/paper-budget
fedvar evaluate --checkpoint runs/mnist-fedvar/checkpoint.pt --output runs/mnist-evaluation.json
```

Use a different output directory for every run. `compare` uses identical seeded datasets, client partitions, model initialization and client sampling across the four methods; their learned model states evolve separately. The synthetic dataset uses common class prototypes and independent train/test noise. Vision datasets use their official train/test split; test labels never inform client selection or aggregation.

For GPU runs, install matching PyTorch and torchvision builds and pass `--device cuda`. Numerical trajectories can vary with device/library versions; CPU tests verify deterministic repeated runs in one environment. Dependency versions are recorded in `metrics.json`.

## Mobile model families

```bash
pip install -e '.[mobile]'
fedvar train --config configs/mnist.json --model ghostnet_100 --download --output runs/ghostnet
```

Supported timm model names are `tinynet_a`, `ghostnet_100`, and `mobilenetv3_small_100`. Models are initialized without pretrained weights. For a singleton local minibatch, batch-normalization layers use their stored running statistics while their affine parameters continue to learn; ordinary minibatches update running statistics normally. This explicit rule supports small Dirichlet clients without dropping examples. The default CNN/MLP contain no batch-normalization layers.

## Artifacts

Each run saves `config.json`, exact train indices and class histograms in `partition.json`, per-round metrics/selection diagnostics in `metrics.json`, a compact `metrics.csv`, and the final server `checkpoint.pt`. `evaluate` reloads that server checkpoint and evaluates its configured test split. Checkpoints support evaluation; training continuation is not exposed by the CLI.

## Validation

```bash
pip install -e '.[dev]'
pytest -q
ruff check .
fedvar train --config configs/smoke.json --output runs/check
fedvar evaluate --checkpoint runs/check/checkpoint.pt
```

Tests cover the analytical norm interval and selected average, zero/equal/boundary cases, tensor alias protection, dtype/shape validation, sample-weighted baseline aggregation, partition conservation, client isolation, FedSGD's gradient equivalence, all four training paths, checkpoint evaluation, and deterministic CPU runs. CI verifies the complete execution path on synthetic data. Vision dataset runs save their measured accuracy and loss in the same artifact format.

## Source context

We acknowledge [c-gabri/Federated-Learning-PyTorch](https://github.com/c-gabri/Federated-Learning-PyTorch), the experimental platform cited in the paper. Publication metadata is available in the [Sungkyunkwan University record](https://pure.skku.edu/en/publications/fedvar-federated-learning-algorithm-with-weight-variation-in-clie/).
