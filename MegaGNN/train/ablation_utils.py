"""
Utilities for the random-grouping ablation experiment.

The intervention: within each batch, for each edge type, randomly permute the
src row of edge_index within each dst group.  edge_attr and e_id stay bound to
their original edge positions.  When cfg.ablation.random_grouping is True,
HeteroMultiEdgeAggregator uses group IDs derived from the shuffled (src, dst)
pairs instead of the precomputed simp_edge_batch.
"""

import torch
from torch_geometric.data import HeteroData

from MegaGNN.graphgym.config import cfg


def random_group_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    """Per-target random permutation of the src row of edge_index.

    edge_attr and any per-edge identifier tensors are NOT modified; they remain
    bound to their original edge positions.

    Args:
        edge_index: LongTensor of shape [2, E], row 0 = src, row 1 = dst.

    Returns:
        new_edge_index: LongTensor of shape [2, E] with src shuffled within
                        each dst group.
    """
    src, dst = edge_index[0], edge_index[1]
    device = src.device
    E = src.size(0)

    if E == 0:
        return edge_index.clone()

    # Sort edges so same-target edges are contiguous.
    order = torch.argsort(dst, stable=True)

    # Within each dst group, break ties with uniform noise → within-group shuffle.
    noise = torch.rand(E, device=device)
    # Scale noise so it never crosses dst boundaries.
    composite = dst[order].to(torch.float64) * (noise.max().item() + 2.0) + noise
    inner = torch.argsort(composite)

    # Inverse of `order` maps back to the original edge positions.
    inv = torch.empty_like(order)
    inv[order] = torch.arange(E, device=device)

    shuffled_src = src[order][inner][inv]
    return torch.stack([shuffled_src, dst], dim=0)


def group_indices_from_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    """Derive group IDs from (src, dst) pairs in edge_index, GPU-friendly.

    Edges sharing the same (src, dst) pair receive the same integer group ID.
    This replaces the precomputed simp_edge_batch when random_grouping is on.

    Args:
        edge_index: LongTensor [2, E].

    Returns:
        group_ids: LongTensor [E], contiguous group indices starting at 0.
    """
    # Pack (src, dst) into a single int64 composite key.
    # Assumes node indices fit in int32 (safe for all AML datasets).
    src, dst = edge_index[0], edge_index[1]
    max_nodes = max(src.max().item(), dst.max().item()) + 1
    composite_key = dst * max_nodes + src
    _, group_ids = torch.unique(composite_key, return_inverse=True)
    return group_ids


def apply_random_group_control(batch: HeteroData) -> HeteroData:
    """Shuffle src labels per dst group for every edge type in the batch.

    Only active when cfg.ablation.random_grouping is True.  Applied on CPU
    before batch.to(device) so that the shuffled edge_index is what the model
    sees.

    After the shuffle, HeteroMultiEdgeAggregator.forward will call
    group_indices_from_edge_index(edge_index) to derive grouping from the
    scrambled (src, dst) pairs.

    Args:
        batch: HeteroData batch from the sampler.

    Returns:
        batch with edge_index[0] shuffled per dst group for each edge type.
    """
    if not cfg.ablation.random_grouping:
        return batch

    for edge_type in batch.edge_types:
        ei = batch[edge_type].edge_index
        batch[edge_type].edge_index = random_group_edge_index(ei)

    return batch


# ---------------------------------------------------------------------------
# Sanity checks (PRD §6, validation step 1)
# ---------------------------------------------------------------------------

def sanity_check_random_grouping(seed: int = 0) -> None:
    """Verify correctness of random_group_edge_index on a toy example.

    Checks:
      (a) dst row is unchanged.
      (b) src row is a permutation of the original within each dst group.
      (c) The global multiset of src values is preserved.

    Raises AssertionError on failure; prints a summary on success.
    """
    torch.manual_seed(seed)

    # Build a toy edge_index with repeated dst nodes (multigraph).
    # dst=0 has 4 edges, dst=1 has 2 edges, dst=2 has 1 edge.
    src = torch.tensor([10, 20, 30, 40, 11, 21, 15])
    dst = torch.tensor([0,  0,  0,  0,  1,  1,  2])
    edge_index = torch.stack([src, dst], dim=0)

    shuffled = random_group_edge_index(edge_index)
    new_src, new_dst = shuffled[0], shuffled[1]

    # (a) dst unchanged
    assert torch.equal(new_dst, dst), "dst row was modified"

    # (b) within each dst group, new_src is a permutation of original src
    for d in dst.unique():
        mask = dst == d
        orig_group = src[mask].sort().values
        new_group = new_src[mask].sort().values
        assert torch.equal(orig_group, new_group), (
            f"Src multiset changed for dst={d.item()}: "
            f"orig={orig_group.tolist()} new={new_group.tolist()}"
        )

    # (c) global src multiset preserved
    assert torch.equal(src.sort().values, new_src.sort().values), \
        "Global src multiset changed"

    print("[sanity_check_random_grouping] All assertions passed.")
    print(f"  Original  src: {src.tolist()}")
    print(f"  Shuffled  src: {new_src.tolist()}")
    print(f"  dst (unchanged): {new_dst.tolist()}")


if __name__ == '__main__':
    sanity_check_random_grouping()
