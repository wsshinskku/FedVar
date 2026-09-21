# FedVar

**English** | [한국어](README.ko.md)

Official author implementation of **FedVar: Federated Learning Algorithm with Weight Variation in Clients**. FedVar filters participating client models by their parameter norms and averages the models inside one population standard deviation of the mean.

**Wooseok Shin and Jitae Shin · ITC-CSCC 2022 · pp. 456–459**

[Paper DOI](https://doi.org/10.1109/ITC-CSCC55581.2022.9894899) · [Paper PDF](FedVar__Federated_Learning_Algorithm_with_Weight_Variation_in_Clients.pdf) · [Algorithm](docs/ALGORITHM.md) · [Experiments](docs/EXPERIMENTS.md)

## Quick start

Python 3.10 or newer is required. This CPU example runs without downloading a dataset.

```bash
git clone https://github.com/wsshinskku/FedVar.git
cd FedVar
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
fedvar train --config configs/smoke.json --output runs/smoke
fedvar evaluate --checkpoint runs/smoke/checkpoint.pt
```

Use `python -m fedvar` if the console command is not on your PATH. The existing filename remains available as `python FedVar.py train --config configs/smoke.json --output runs/legacy-entry` after installation.

## MNIST and baseline comparison

```bash
python -m pip install -e '.[vision]'
fedvar train --config configs/mnist.json --download --output runs/mnist
fedvar compare --config configs/mnist.json --download --output runs/comparison
```

The comparison runs **FedAvg, FedSGD, FedProx, and FedVar** with the same seeded initialization, dataset partition, and participating-client schedule. Supported image datasets are MNIST, CIFAR-10, and CIFAR-100; select one with `--dataset cifar10`. Dataset downloads require `--download`. Select a compatible CUDA device with `--device cuda`.

The built-in CNN and MLP are self-contained. The optional `mobile` extra supplies TinyNet, GhostNet, and MobileNetV3 families through timm; see [model choices and batch-normalization considerations](docs/EXPERIMENTS.md#mobile-model-families).

## Method

For a participating client's flattened trainable parameters `w_k`, let `z_k = ||w_k||_2`:

```text
mean = average(z_k)
std  = sqrt(average((z_k - mean)^2))
I    = {k : mean - std <= z_k <= mean + std}
next_server_model = average(w_k for k in I)
```

This follows the norm comparison described in the paper and the inclusive population-standard-deviation filter of Algorithm 2. Norms select models; actual model tensors are averaged. FedVar assigns equal weights to retained clients. The FedAvg comparator assigns weights proportional to client sample counts. Every round trains isolated copies of the current server model and evaluates the server **after** aggregation on a separate test split.

The [algorithm guide](docs/ALGORITHM.md) explains tensor buffers, boundary cases, baseline semantics, and IID/Dirichlet/label-shard partitioning.

## Configurations and outputs

| Configuration | Purpose |
|---|---|
| `configs/smoke.json` | Fast synthetic-data end-to-end check |
| `configs/mnist.json` | Practical MNIST experiment: 20 clients, 20 rounds |
| `configs/paper-budget-mnist.json` | Published budget: 100 clients, 10 participants, 200 rounds, 5 local epochs, batch 50, evaluation every 20 rounds |

The final preset uses MNIST, the included CNN, and a Dirichlet-0.5 partition with the published training budget. Every run records its resolved architecture, optimizer, random seed, and exact client partition so experiments can be inspected and repeated.

Each run saves:

- `config.json`: all resolved settings.
- `partition.json`: exact client training indices and class histograms.
- `metrics.json`: accuracy/loss, participants, retained clients, model norms and aggregation weights.
- `metrics.csv`: per-round metrics for plotting.
- `checkpoint.pt`: final server model for the `evaluate` command.

Use a new or empty output directory for each experiment. Training continuation is not exposed by this CLI; checkpoints support evaluation.

## Repository layout and checks

```text
src/fedvar/        Aggregation, data, models, training, and CLI
configs/           Runnable experiment settings
tests/             Algorithm, training, and checkpoint checks
docs/              Paper-to-code details and experiment guidance
legacy/            Unmodified earlier FedVar.py source
```

```bash
pytest -q
ruff check .
```

The maintained package corrects the earlier norm-ratio sum, shared client model references, mutation of the first client's tensors, missing server aggregation, and undefined model/constructor arguments. The original source remains unchanged in [legacy/FedVar_original.py](legacy/FedVar_original.py). The paper PDF is preserved at its existing path.

## Citation

```bibtex
@inproceedings{shin2022fedvar,
  author = {Shin, Wooseok and Shin, Jitae},
  title = {FedVar: Federated Learning Algorithm with Weight Variation in Clients},
  booktitle = {2022 37th International Technical Conference on Circuits/Systems,
               Computers and Communications (ITC-CSCC)},
  year = {2022},
  pages = {456--459},
  doi = {10.1109/ITC-CSCC55581.2022.9894899}
}
```

Machine-readable citation information is available in [CITATION.cff](CITATION.cff). We acknowledge [c-gabri/Federated-Learning-PyTorch](https://github.com/c-gabri/Federated-Learning-PyTorch), the experimental platform cited in the paper.
