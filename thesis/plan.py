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


def build_plan():
    """Ordered list of runs (dataset, model, config, seed).

    Wave 1: baseline and full, per seed, LI then HI (pairs complete early).
    Wave 2: ablations, per seed, LI then HI.
    """
    plan = []
    for ds in DATASETS:
        for seed in MAIN_SEEDS:
            for cfg in MAIN_CONFIGS:
                plan.append((ds, BACKBONE, cfg, seed))
    for ds in DATASETS:
        for seed in ABLATION_SEEDS:
            for cfg in ABLATION_CONFIGS:
                plan.append((ds, BACKBONE, cfg, seed))
    return plan


def estimate_hours(key):
    return EST_H[key[0]]


N_RUNS = len(DATASETS) * (len(MAIN_CONFIGS) * len(MAIN_SEEDS) + len(ABLATION_CONFIGS) * len(ABLATION_SEEDS))
assert len(build_plan()) == N_RUNS == 70
