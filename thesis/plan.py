r"""Run plan of the thesis main experiment, read by Eksperimen_Utama_Kaggle.ipynb.

Kept in the repository so the plan can change without re-importing the
notebook on Kaggle (each session clones the latest commit).

Design (proposal, Opsi B; supervisor decision 2026-10-21: backbone Multi-GIN+EU only):
five configurations on Multi-GIN+EU x two datasets; baseline and full with
seeds 1-10 (paired, for the Wilcoxon test of H1), the three ablations with
seeds 1-5 -> 2 x (10 + 10 + 5 + 5 + 5) = 70 runs.
"""

BACKBONE = "Multi-GIN"
DATASETS = ["Small-LI", "Small-HI"]          # LI first: H1 is tested on LI

# Command-line overrides of cfg.temporal per configuration (name_tag = key).
OVERRIDES = {
    "base": [],
    "t2v": ["temporal.encoding", "time2vec"],
    "sin": ["temporal.encoding", "sinusoid"],
    "decay": ["temporal.decay", "True"],
    "full": ["temporal.encoding", "time2vec", "temporal.decay", "True"],
}

MAIN_CONFIGS, MAIN_SEEDS = ["base", "full"], range(1, 11)
ABLATION_CONFIGS, ABLATION_SEEDS = ["t2v", "sin", "decay"], range(1, 6)

# Conservative duration of one run (hours) with two runs in parallel on Kaggle
# T4 x2, at the maximum number of epochs, including ~10% for the temporal module
# (pilot: HI ~311 s/epoch, LI ~448 s/epoch sequential; parallel ~1.12x slower;
# HI at most 20 epochs, LI at most 40).
EST_H = {"Small-HI": 2.2, "Small-LI": 6.3}


# Datasets whose ablation runs go first (decision 2026-10-26). After session 3 the
# full configuration started learning several epochs later than the baseline
# even with the zero-initialised encoding projection; the HI ablations (cheap,
# ~2 h each) separate the encoding from the decay component.
ABLATIONS_FIRST = ["Small-HI"]

# Configurations queued in the current phase (None = all). Decision 2026-10-27:
# the HI ablations showed that the delta_t encoding generalises poorly because
# delta_t is left-censored by the start of the data in the training period; a
# delta_t horizon is under discussion with the supervisor. Baselines do not use
# delta_t, so only they are run until that is decided.
ACTIVE_CONFIGS = ["base"]


def full_plan():
    """All 70 runs (dataset, model, config, seed), in queue order.

    0. Ablations of the datasets in ABLATIONS_FIRST, per seed.
    1. Baseline and full, per seed, LI then HI (pairs complete early).
    2. Remaining ablations, per seed.
    """
    plan = []
    for ds in ABLATIONS_FIRST:
        for seed in ABLATION_SEEDS:
            for cfg in ABLATION_CONFIGS:
                plan.append((ds, BACKBONE, cfg, seed))
    for ds in DATASETS:
        for seed in MAIN_SEEDS:
            for cfg in MAIN_CONFIGS:
                plan.append((ds, BACKBONE, cfg, seed))
    for ds in DATASETS:
        if ds in ABLATIONS_FIRST:
            continue
        for seed in ABLATION_SEEDS:
            for cfg in ABLATION_CONFIGS:
                plan.append((ds, BACKBONE, cfg, seed))
    return plan


def build_plan():
    """Runs queued in the current phase (the full plan filtered by ACTIVE_CONFIGS)."""
    return [k for k in full_plan() if ACTIVE_CONFIGS is None or k[2] in ACTIVE_CONFIGS]


def estimate_hours(key):
    return EST_H[key[0]]


N_RUNS_TOTAL = len(DATASETS) * (len(MAIN_CONFIGS) * len(MAIN_SEEDS) + len(ABLATION_CONFIGS) * len(ABLATION_SEEDS))
assert len(set(full_plan())) == len(full_plan()) == N_RUNS_TOTAL == 70
# Runs in the current phase; the notebook reports progress against this number.
N_RUNS = len(build_plan())
