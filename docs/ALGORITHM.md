# Algorithm and implementation

This repository contains the official author implementation of **FedVar**. The maintained aggregation follows Section II and Algorithm 2 on page 458 of the [bundled paper](../FedVar__Federated_Learning_Algorithm_with_Weight_Variation_in_Clients.pdf).

## FedVar aggregation

For the `K` participating local models, flatten each model's trainable parameters into `w_k` and compute:

```text
z_k = ||w_k||_2
mean = sum(z_k) / K
std = sqrt(sum((z_k - mean)^2) / K)
I = { k : mean - std <= z_k <= mean + std }
w_next = sum(w_k for k in I) / |I|
```

The paper's introduction specifies norms for tensor comparison; Algorithm 2 specifies the inclusive one-standard-deviation interval and `SDA(w) = sum(wsd) / n`. Norms select clients; the actual tensors of those selected clients are averaged. The population denominator is `K`, not `K-1`. FedVar uses uniform averaging of retained clients, even when their dataset sizes differ. The standard FedAvg comparator uses sample-count weights.

Each round starts selected clients from independent copies of the current server model. After local SGD, the server applies the filter, updates its own model, and evaluates it on the separate test set. A client excluded in one round remains eligible for sampling in later rounds. Client sampling and the norm filter are distinct steps; logs expose both `participants` and `retained_clients`.

In a finite set, at least one norm is within one population standard deviation of the mean. Identical norms, including zero norms, keep all clients. Eight floating-point epsilons at the interval boundary handle numerical roundoff; there is no additional statistical threshold. The implementation rejects nonfinite tensors and never divides a model by the mean norm.

The runner passes `named_parameters()` keys to the norm computation. Floating buffers, if present, are averaged over the retained clients. Nonfloating counters/flags are copied from the first retained client. The included CNN/MLP contain no running-statistic buffers. Optional mobile architectures can contain batch normalization: singleton minibatches use stored running statistics, while ordinary minibatches update them. Every training example is retained.

## Baselines

| Method | Local operation | Server operation |
|---|---|---|
| FedVar | Local minibatch SGD for `local_epochs` | Uniform average of norm-filtered clients |
| FedAvg | Same local SGD | Sample-count-weighted average of all participants |
| FedProx | SGD on cross entropy + `mu/2 * ||w-w_global||^2` | Sample-count-weighted average |
| FedSGD | One gradient step using the complete local dataset | Sample-count-weighted average |

FedSGD accumulates sample-normalized gradients across batches before one optimizer step; `local_epochs` is not used for this baseline. With the provided CNN/MLP, this is equivalent to a single gradient step over the participating clients' pooled data. Optimizers restart at every round. Momentum and weight decay are explicit configuration values.

## Client distributions

- `iid`: seeded shuffle followed by approximately equal contiguous allocations.
- `dirichlet`: draw class-specific client proportions with concentration `dirichlet_alpha`, then multinomial counts. Empty clients receive one existing example from the largest partition. No samples are duplicated or discarded.
- `shards`: shuffle within labels, sort by label, form `clients * shards_per_client` shards, and distribute a fixed number of shuffled shards per client.

The paper uses `s=0, 0.5, 1` as IID/semi-IID/non-IID labels. It does not specify an exact conversion from `s` to a partition algorithm or Dirichlet concentration. Consequently this implementation records the actual split recipe and indices instead of assigning an undocumented `s` mapping.

## Maintenance changes

The earlier `FedVar.py` is preserved unchanged in [legacy/FedVar_original.py](../legacy/FedVar_original.py). The maintained package repairs the old norm-ratio aggregation, shared client model references, in-place state mutation, missing server aggregation call, undefined model/constructor arguments, and missing dataset integration. `FedVar.py` now forwards to the packaged CLI.
