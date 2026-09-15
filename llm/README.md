# llm/

Federated LoRA fine-tuning of language models for NF-ToN-IoT attack
classification, with a model-agnostic pipeline so BERT, T5, Llama, Qwen and
Gemma all run through identical code, loss and metrics.

All experiments run on **Kaggle** (T4 x2, 30h/week, 9h max per session). Nothing
in this pipeline is meant to run on a laptop.

```
llm/
├── split_dataset.py     # step 1: build client shards + shared val/test
├── check_models.py      # step 2: verify model ids before spending GPU quota
├── run_centralized.py   # baseline: pooled training, no federation
├── run_experiment.py    # the federated run
├── configs/
│   ├── base.yaml            # shared settings
│   └── models/*.yaml        # one file per model - the only thing that differs
└── fedllm/
    ├── config.py        # base + model config merge
    ├── data.py          # row -> text, client shards, per-round slices
    ├── modeling.py      # SEQ_CLS + LoRA for every architecture
    ├── client.py        # local training loop
    ├── aggregation.py   # naive / delta_svd / ffa / performance
    ├── server.py        # federated round loop + checkpointing
    └── evaluate.py      # metrics
```

## How the pipeline works

**Two levels of data.** Each client owns a *shard* — a fixed, private slice of
the training pool that never mixes with any other client's. Each round, a client
trains on a small *slice* of its own shard (`rows_per_client_per_round`), walking
through it in order. So the shard stays large and realistic while per-round
compute stays affordable, and a client never sees the same row twice.

**Only LoRA travels.** Clients send back their LoRA matrices plus the
classification head — a few MB — never raw rows. The head has to be included:
it starts from a random init, so averaging LoRA while leaving five different
heads in place would produce a meaningless global model.

**Aggregation is the research question.** A LoRA update is a product `dW = B @ A`,
and `mean(B_i @ A_i) != mean(B_i) @ mean(A_i)`. The four strategies in
`aggregation.py` are different answers to that, and comparing them is the point:

| Strategy | Idea | Exact? |
|---|---|---|
| `naive` | average A and B separately (FedIT) | no — measurably off |
| `delta_svd` | average the real `B @ A`, refactor via SVD | best rank-r approximation |
| `ffa` | freeze A, train and average only B | yes, and halves upload |
| `performance` | weight by validation score, not sample count | same as naive, different weights |

## Running it on Kaggle

Attach the NF-ToN-IoT dataset to the notebook, then build the shards in the same
session. Nothing needs to be prepared locally.

### Cell 1 — setup

```python
!pip install -q -U transformers peft accelerate comet_ml
!git clone -q https://github.com/<you>/Fed_GNN.git /kaggle/working/repo
%cd /kaggle/working/repo

import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")  # write token, for adapter upload
```

Gemma and Llama are gated — accept their licences on the Hub first, or the
download returns 403. Enable **Internet** in the notebook settings (needed for
the Hub and for Comet).

### Cell 2 — build the shards from the attached dataset

```python
!python llm/split_dataset.py \
    --input_file /kaggle/input/nf-ton-iot/NF-ToN-IoT.csv \
    --output_dir /kaggle/working/dataset \
    --num_clients 5 --alpha 0.5
```
Adjust the input path to match the attached dataset. Add `--max_rows` only for a
quick trial run; the full 1.38M rows are fine, since per-round sampling is what
controls cost.

### Cell 3 — verify the models before spending quota

```python
!python llm/check_models.py --load
```
Confirms each model id still resolves and that its architecture supports
`AutoModelForSequenceClassification` — the assumption the whole pipeline rests on.

### Cell 4 — run both splits in parallel, one per T4

Each `!` line is its own shell, so `!cmd &` followed by `!wait` does not actually
wait for anything. Launch both from Python instead:

```python
import subprocess, os

procs = []
for gpu, split in [(0, 'iid'), (1, 'non_iid')]:
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}
    log = open(f'/kaggle/working/{split}.log', 'w')
    procs.append(subprocess.Popen([
        'python', 'llm/run_experiment.py', '--model', 'qwen', '--split', split,
        '--data_dir', '/kaggle/working/dataset', '--output_dir', '/kaggle/working/results',
    ], env=env, stdout=log, stderr=subprocess.STDOUT))

for p in procs:
    p.wait()
```

Kaggle bills session wall-clock, not GPU count, so running both splits together
gets two experiments per quota hour. Every round is checkpointed to the output
dir — if a session dies, rerun the same command with `--resume`.

The final LoRA adapter (a few MB) uploads to your Hub account automatically as
`fed-ids-<model-full-name>` — e.g. `fed-ids-Qwen3.5-2B`. Pass `--no_push` to skip
it, and note that `HF_TOKEN` must have write access or the upload is skipped with
a warning.

Because the repo name keys off the model alone, every run of the same model
(each split, each aggregation strategy) pushes to the same repo and overwrites
the previous adapter. `results.json` and Comet keep the full record either way,
but if you want each run's weights kept separately, say so and the run suffix
goes back into the name.

## Experiment tracking

Every run logs to Comet ML under the project **`fed-llm-ids`**, tagged with model,
split and aggregation strategy so runs can be filtered in the UI.

- **Parameters:** model id, LoRA config, federated schedule, training
  hyperparameters, plus the full dataset provenance read from `split_config.json`
  (source file, row counts, Dirichlet alpha, per-client shard sizes). Any result
  can be traced back to the partition that produced it.
- **Per round:** mean and per-client loss, round duration, upload MB per client,
  and the validation metrics.
- **At the end:** test accuracy, balanced accuracy, macro/weighted F1, per-class
  F1, the confusion matrix, total wall-clock and total communication cost.

Pass `--no_comet` to turn tracking off. Comet failures are caught and logged —
they never abort a run.

### Baselines

```bash
python llm/run_centralized.py --model bert --split iid    # upper bound
```
The centralized baseline consumes exactly the same number of rows as a federated
run (`clients x rows_per_round x rounds`), so the gap between them is the cost of
federation and nothing else.

## Keeping the comparison honest

- **Same data budget for every model.** `rows_per_client_per_round` is set by the
  slowest model (the 2B ones). Do not give BERT more just because it is cheap.
- **Same held-out data.** `val.csv` and `test.csv` are shared across both splits
  and all models.
- **No label leakage.** The serializer drops `Label` (the binary flag) and
  `Attack` (the target). It also drops source/destination IPs by default —
  attacker addresses are nearly unique per class in NF-ToN-IoT, so keeping them
  lets a model memorise addresses instead of learning traffic behaviour. Set
  `include_ips: true` in `base.yaml` to study that effect deliberately.
- **Do not class-balance the per-round sample.** Sampling must preserve each
  client's own distribution, otherwise the non-IID skew you are measuring
  disappears and both splits give the same answer.

## Timing on one T4

At `rows_per_client_per_round: 2000`, 5 clients, 10 rounds:

| Model | Params | Approx. per run |
|---|---|---|
| bert-base-uncased | 110M | ~20 min |
| t5-base | 220M | ~45 min |
| Llama-3.2-1B | 1B | ~2.5 h |
| Qwen3.5-2B | 2B | ~5 h |
| gemma-4-E2B | ~2B | ~5 h |

These are FLOP-based estimates, easily +/-40% off. The server logs a projected
total after the first round — trust that number over this table.

## split_dataset.py

Holds out a shared validation and test set, then splits the remaining training
pool into per-client shards for a number of simulated federated clients,
producing two variants of that partition side by side:

- **IID** — every client gets a stratified random sample, so each client sees
  roughly the same attack-type distribution.
- **Non-IID** — a Dirichlet(alpha) label-skew split, so clients end up with
  very different attack-type distributions. Lower `--alpha` = more skew,
  higher `--alpha` = closer to IID.

### Output layout

```
llm/dataset/
  ├── val.csv                   # shared across both variants
  ├── test.csv                  # shared across both variants
  ├── split_distribution.csv
  ├── iid/
  │   ├── client_1.csv
  │   ├── ...
  │   ├── client_N.csv
  │   └── attack_distribution.csv
  └── non_iid/
      ├── client_1.csv
      ├── ...
      ├── client_N.csv
      └── attack_distribution.csv
```

`val.csv` and `test.csv` sit at the top level on purpose — every model and
every partitioning strategy must be evaluated on the exact same held-out data,
otherwise the numbers are not comparable. The holdout is stratified, so each
split keeps the original attack-type proportions.

Each run logs three tables to stdout: the global train/val/test distribution,
then one table per variant (rows = attack type, columns = client) showing how
many rows of each attack type landed on each client.

### Usage

Activate the project virtualenv first:

```bash
source .venv/bin/activate
```

Then run the script from the repo root:

```bash
python llm/split_dataset.py --input_file datasets/nftoniot/NF-ToN-IoT.csv --num_clients 5 --max_rows 100000 --alpha 0.5
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--input_file` | *(required)* | Path to the raw CSV dataset |
| `--output_dir` | `llm/dataset` | Where to write the `iid/` and `non_iid/` folders |
| `--num_clients` | `5` | Number of federated clients to split into |
| `--max_rows` | `None` | Randomly downsample the dataset to at most this many rows before splitting |
| `--test_ratio` | `0.1` | Fraction held out as the shared test set |
| `--val_ratio` | `0.1` | Fraction held out as the shared validation set |
| `--label_col` | `Attack` | Column used for the attack-type distribution and the non-IID skew |
| `--alpha` | `0.5` | Dirichlet concentration for the non-IID split (lower = more skewed) |
| `--seed` | `42` | Random seed for reproducibility |

### Examples

Small quick test run (2k rows, 4 clients, strong skew):

```bash
python llm/split_dataset.py --input_file datasets/nftoniot/NF-ToN-IoT.csv --num_clients 4 --max_rows 2000 --alpha 0.3
```

Different source dataset:

```bash
python llm/split_dataset.py --input_file datasets/cictoniot/CIC-ToN-IoT.csv --num_clients 5 --max_rows 100000 --alpha 0.5
```
