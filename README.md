# Can Model Merging Improve Aggregation in DiLoCo?

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b.svg)](#citation)

Stefan Horoi, Benjamin Thérien, Guy Wolf, Eugene Belilovsky

This repository contains the code accompanying the paper **"Can Model Merging Improve Aggregation in DiLoCo?"** (arXiv link forthcoming). It implements model-merging-based outer aggregation strategies for DiLoCo — including IsoLoCo (`ISOCLoCo`), `TSVLoCo`, `ISOCTSLoCo`, and `MergeLoCo` — along with the DiLoCo and merging baselines used in the paper.

The code is adapted from the open-source [SparseLoCo](https://github.com/tplr-ai/SparseLoCo) codebase, which provides the underlying distributed-training scaffolding (sharded data pipeline, training loop, baseline strategies). All credit for that scaffolding goes to the original authors.

## Getting Started

### 1. Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for environment management
- Tested with H100 and H200 GPUs

### 2. Installation

Clone the repository, then install the dependencies with `uv`:

```bash
git clone https://github.com/shoroi/isoloco.git
cd isoloco
uv sync
source .venv/bin/activate
```

### 3. Data Preparation

The training script expects a pre-tokenized and sharded dataset. Use `pretokenize_data.py` to process a dataset from Hugging Face. The default configuration uses `mlfoundations/dclm-baseline-1.0-parquet`. The sweep configs expect the tokenized output at `$DATA_DIR/dclm_tokenized-train_valid_test`.

```bash
export DATA_DIR="~/datasets"
python pretokenize_data.py --output_dir $DATA_DIR/dclm_tokenized-train_valid_test --total_tokens 10e9
```

## Running Experiments

Experiments are managed via `wandb` sweeps. First, set your W&B API key:

```bash
export WANDB_API_KEY="..."
```

Each sweep is configured to run on 4 GPUs by default (`--nproc_per_node=4`); adjust this in the sweep YAML if your hardware differs.

### Example — IsoLoCo at 178M, replication factor 8

```bash
wandb sweep hparams/178M/sweeps_main/r8_sweep_isoloco.yaml
# the previous command prints a SWEEP_ID; pass it to the agent:
wandb agent $SWEEP_ID
```

Additional sweep configurations (other strategies, larger model size, merging baselines) are available under `hparams/178M/sweeps_main/`, `hparams/178M/sweeps_merging/`, and `hparams/512M/sweeps_main/`. Run any of them with the same `wandb sweep` / `wandb agent` pattern shown above.

### Running on a SLURM cluster

Example launch scripts are provided in `scripts/` and can be submitted with a sweep ID:

```bash
sbatch scripts/run_sweep_id_12h.sh $SWEEP_ID
```

These scripts contain our cluster's specifics (SLURM account, partitions/GPUs, module loads, and `SAVE_DIR`/`DATA_DIR` paths) and must be adapted to your own SLURM environment before use.

## Citation

If you find this work useful, please cite:

```bibtex
@article{horoi2026merging,
  title  = {Can Model Merging Improve Aggregation in DiLoCo?},
  author = {Horoi, Stefan and Thérien, Benjamin and Wolf, Guy and Belilovsky, Eugene},
  year   = {2026},
  note   = {Preprint. arXiv link forthcoming.}
}
```
