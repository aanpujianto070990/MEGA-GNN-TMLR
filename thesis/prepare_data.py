r"""Build the processed AML cache once, before starting several runs at the same time.

Runs started in parallel on an empty cache would all process the dataset and
write the same files concurrently. Building it once up front avoids that.

Usage (from the repository root):
  python thesis/prepare_data.py --dir /tmp/aml_data --name Small-LI
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import MegaGNN  # noqa: E402,F401  (registers all config options)
from MegaGNN.graphgym.config import cfg, set_cfg  # noqa: E402
from MegaGNN.datasets.aml_dataset import AMLDataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="dataset.dir (contains AML/<name>/raw)")
    parser.add_argument("--name", required=True, help="Small-HI or Small-LI")
    args = parser.parse_args()

    set_cfg(cfg)
    t0 = time.time()
    # process() builds every cached file (ports, parallel-edge groups, ...)
    # regardless of these flags; they only select what is loaded afterwards.
    dataset = AMLDataset(root=os.path.join(args.dir, "AML"), name=args.name,
                         reverse_mp=True, add_ports=True, multi_edge_agg=True)
    test = dataset.data_dict["test"]["node", "to", "node"]
    print(f"{args.name}: cache ready in {time.time() - t0:.0f}s | "
          f"edges in test graph {test.edge_index.shape[1]} | "
          f"has edge_id {'edge_id' in test} | has delta_t {'delta_t' in test}")


if __name__ == "__main__":
    main()
