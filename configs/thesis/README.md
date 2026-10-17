# Thesis configurations

Base configs: `AML-Small-{HI,LI}-{Multi,MEGA}-GIN.yaml` (paper model and optimizer
settings including batch_accumulation 4, thesis training protocol: HI at most 20
epochs with early-stopping patience 5, LI at most 40 epochs with patience 8, because
in the pilot both LI baselines were still improving after 20 epochs). The experiment configurations are the base
config plus command-line overrides of `cfg.temporal` (MegaGNN/config/thesis_config.py,
MegaGNN/layer/temporal.py):

| Configuration | Overrides | Backbones | Seeds |
|---|---|---|---|
| baseline | (none) | Multi-GIN, MEGA-GIN | 1-10 |
| + Time2Vec | `temporal.encoding time2vec` | Multi-GIN | 1-5 |
| + fixed sinusoid | `temporal.encoding sinusoid` | Multi-GIN | 1-5 |
| + decay | `temporal.decay True` | Multi-GIN | 1-5 |
| full (Time2Vec + decay) | `temporal.encoding time2vec temporal.decay True` | Multi-GIN, MEGA-GIN | 1-10 |

Example:

```
python -m MegaGNN.main --cfg configs/thesis/AML-Small-LI-Multi-GIN.yaml --gpu 0 \
    seed 3 name_tag full temporal.encoding time2vec temporal.decay True \
    out_dir results dataset.dir data
```

Temporal module, with delta_t = seconds since the sender's previous incoming
transaction (strictly earlier; -1 if none), used in days:
- encoding: Time2Vec [w0 t + p0, sin(wi t + pi)], k = `temporal.dim` periodic
  components initialised at periods log-spaced between `temporal.min_period_days`
  and `temporal.max_period_days`, plus a has-history flag, projected and added to
  the encoded edge features; `sinusoid` keeps the same initial values frozen;
- decay: w = exp(-lambda t) (w = 1 without history), lambda learned from
  `temporal.decay_init`; multiplies each GINE message before the neighbour sum
  (Multi-GIN) or each parallel edge before the parallel-edge sum (MEGA-GIN).

`run_summary.json` records the learned lambda and Time2Vec periods.
