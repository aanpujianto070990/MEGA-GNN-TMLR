# MegaGNN

> **Fork for a master's thesis** (temporal module on Multi-GIN+EU and MEGA-GIN, AMLworld HI/LI-Small).
> Changes vs. upstream so far are compatibility-only (no model or protocol changes):
> a vectorized pandas `format_dataset` (no `datatable` dependency, same ids and ordering),
> `torch.load(..., weights_only=False)` for processed data and checkpoints (needed on `torch>=2.6`),
> an optional `datatable` import in `jodie_dataset.py`, and removal of an unused `scipy.stats.stats` import.
> `Pilot_Reproduksi_Kaggle.ipynb` installs everything on Kaggle and reproduces Multi-GIN / MEGA-GIN.
>
> Thesis training protocol (`configs/thesis/`, opt-in via `train.test_at_best`; upstream configs behave as before):
> full-pass epochs (max. 20), validation every epoch, model selection and early stopping on validation F1,
> a single test evaluation with the best checkpoint, `run_summary.json` (timings, peak GPU memory, coverage)
> and per-transaction scores (`scores_val.npz`, `scores_test.npz`, with the raw CSV row as `edge_id`).
> `Uji_Protokol_Tesis_Kaggle.ipynb` runs the seed-1 baselines with this protocol on Kaggle;
> `Eksperimen_Utama_Kaggle.ipynb` runs the 70-run main experiment on Multi-GIN+EU (`thesis/plan.py`)
> two at a time on Kaggle T4 x2, resuming across sessions.

## Setup

- Create a new Conda environment, and activate.
```bash
conda env create -f env.yaml
conda activate megagnn
```
- Install Pytorch and Pytorch Geometric
```bash
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip install -r requirements.txt
```
- Install genagg and datatable
```bash
cd genagg
pip install -e .
cd ..
pip install datatable
```
- Build the `ports_cpp` extension, which the dataloaders use to compute port
  numberings:
```bash
cd MegaGNN/cpp
python setup.py build_ext --inplace
cd ../..
```


## Datasets

### Anti-Money Laundering (AML)

Download the transaction files from
[Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data)
and place the raw `*_Trans.csv` file in the `raw` directory of the corresponding
dataset (see the layout below). No manual pre-processing is needed: on the first
run the dataloader converts the raw file into `formatted_transactions.csv` and
caches it next to the raw data.

### Ethereum Phishing Detection (ETH)

Download `edges-kaggle.csv` and `nodes-kaggle.csv` from
[ETH-Kaggle](https://drive.google.com/drive/folders/1u-NZ96U1SObxXEdWuClOufInbv_5vB1g?usp=sharing)
and place them in `ETH/Kaggle/raw`.

### Directory layout

Point `dataset.dir` at a directory organised as follows. Only the `raw` files
are required — each `processed` directory is created automatically on first use
and reused afterwards.

```
/path/to/data/
├── AML
│   ├── Small-HI
│   │   └── raw
│   │       └── HI-Small_Trans.csv
│   └── Medium-HI
│       └── raw
│           └── HI-Medium_Trans.csv
└── ETH
    └── Kaggle
        └── raw
            ├── edges-kaggle.csv
            └── nodes-kaggle.csv
```

The AML dataset name selects the raw file: `Small-HI` → `HI-Small_Trans.csv`,
`Small-LI` → `LI-Small_Trans.csv`, and likewise for the `Medium-*` and `Large-*`
variants.

### Pointing the configs at your data

Every config sets the data and output directories at the top:

```yaml
out_dir: /path/to/output
dataset:
  dir: /path/to/data
  format: AML     # Which benchmark to use (AML, ETH, ...)
  name: Small-HI  # Which dataset to use
```

Edit these, or override them on the command line without touching the file:

```bash
python -m MegaGNN.main --cfg configs/PNA/AML-Small-HI-MEGA-PNA.yaml \
  dataset.dir /path/to/data \
  out_dir ./results
```

## Usage

### Configuration

MegaGNN uses YAML configuration files located in the `configs/` directory. These files define model architecture, training parameters, and dataset settings.

Example configuration:

```yaml
model:
  type: MegaGNNModel
  loss_fun: weighted_cross_entropy
  loss_fun_weight: [1, 7]
gnn:
  layer_type: PNA  # GINE, GenAgg, PNA, RGCN, RGCNE
  act: relu
  dropout: 0.08
  layers_mp: 2
  dim_inner: 20
  edge_updates: True
  multi_edge_agg: True
  # How parallel edges are aggregated. One of: identity, sum, gin, pna,
  # adamm, transformer, gru, genagg, gt_aggregator
  multi_edge_agg_type: pna
```

### Training

Run a single config directly:

```bash
python -m MegaGNN.main --cfg configs/PNA/AML-Small-HI-MEGA-PNA.yaml
```

Or use `run.sh`, which repeats each config over a number of random seeds:

```bash
./run.sh configs/PNA/AML-Small-HI-MEGA-PNA.yaml

# Several configs, 5 random seeds each
RUNS_PER_CONFIG=5 ./run.sh configs/PNA/AML-Small-HI-MEGA-PNA.yaml configs/ETH/ETH-MEGA-PNA.yaml
```

### Custom Configurations

To create a custom configuration:

1. Copy an existing configuration file from `configs/`
2. Modify parameters as needed
3. Run with your new configuration file


## Key Components

- `MegaGNN/main.py`: Entry point for training and evaluation
- `MegaGNN/network/megagnn.py`: Core GNN model implementation
- `MegaGNN/datasets/`: Dataset implementations for various data sources
- `MegaGNN/layer/`: Custom GNN layer implementations
- `configs/`: Configuration files for different model variants and datasets

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

It builds on third-party code that remains under its own license, most notably
[GraphGym](https://github.com/pyg-team/pytorch_geometric) (MIT) under
`MegaGNN/graphgym/`, and a vendored copy of
[GenAgg](https://github.com/Acciorocketships/generalised-aggregation) under
`genagg/`. See [NOTICE](NOTICE) for the full attribution.

