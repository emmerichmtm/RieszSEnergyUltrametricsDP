from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Node:
    """Rooted binary ultrametric tree node.

    For a leaf, `leaf` is an integer label and `height = 0`.
    For an internal node, `height > 0` is the merge height and
    distances are defined by d(x,y) = 2 * height(lca(x,y)).
    """

    height: float
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    leaf: Optional[int] = None

    @property
    def is_leaf(self) -> bool:
        return self.leaf is not None


# ---------------------------------------------------------------------------
# Tree constructors
# ---------------------------------------------------------------------------

def make_leaf(label: int) -> Node:
    return Node(height=0.0, leaf=label)


def make_internal(left: Node, right: Node, height: float) -> Node:
    if height <= max(left.height, right.height):
        raise ValueError("Internal-node height must exceed child heights.")
    return Node(height=height, left=left, right=right, leaf=None)


# ---------------------------------------------------------------------------
# Basic tree utilities
# ---------------------------------------------------------------------------

def leaves(node: Node) -> Tuple[int, ...]:
    if node.is_leaf:
        return (node.leaf,)  # type: ignore[return-value]
    assert node.left is not None and node.right is not None
    return tuple(sorted(leaves(node.left) + leaves(node.right)))


@lru_cache(maxsize=None)
def leaf_set(node: Node) -> frozenset[int]:
    return frozenset(leaves(node))


def delta_u(node: Node) -> float:
    """Common cross-subtree distance at internal node u."""
    if node.is_leaf:
        raise ValueError("delta_u is defined only for internal nodes.")
    return 2.0 * node.height


def lca_height(node: Node, a: int, b: int) -> float:
    if node.is_leaf:
        return 0.0
    assert node.left is not None and node.right is not None
    left_ids = leaf_set(node.left)
    right_ids = leaf_set(node.right)
    if (a in left_ids and b in right_ids) or (a in right_ids and b in left_ids):
        return node.height
    if a in left_ids and b in left_ids:
        return lca_height(node.left, a, b)
    return lca_height(node.right, a, b)


def ultrametric_distance(root: Node, a: int, b: int) -> float:
    if a == b:
        return 0.0
    return 2.0 * lca_height(root, a, b)


# ---------------------------------------------------------------------------
# Riesz energy and brute force
# ---------------------------------------------------------------------------

def riesz_energy(root: Node, subset: Sequence[int], s: float) -> float:
    total = 0.0
    for a, b in itertools.combinations(subset, 2):
        total += ultrametric_distance(root, a, b) ** (-s)
    return total


def brute_force_best(root: Node, k: int, s: float) -> Tuple[float, Tuple[int, ...]]:
    all_leaves = leaves(root)
    best_energy = math.inf
    best_subset: Tuple[int, ...] = ()
    for subset in itertools.combinations(all_leaves, k):
        val = riesz_energy(root, subset, s)
        if val < best_energy - 1e-15:
            best_energy = val
            best_subset = tuple(subset)
    return best_energy, best_subset


# ---------------------------------------------------------------------------
# Dynamic programming based on the paper pseudocode / notation
# ---------------------------------------------------------------------------

def dp_table(root: Node, s: float) -> Dict[Node, List[float]]:
    """Compute all DP values F_u(t).

    Returns a dictionary mapping each node u to a list F_u where F_u[t]
    is the minimum Riesz s-energy among all t-subsets chosen below u.
    """
    table: Dict[Node, List[float]] = {}

    def visit(u: Node) -> List[float]:
        if u.is_leaf:
            F_u = [0.0, 0.0]
            table[u] = F_u
            return F_u

        assert u.left is not None and u.right is not None
        F_v = visit(u.left)
        F_w = visit(u.right)
        n_v = len(F_v) - 1
        n_w = len(F_w) - 1
        F_u = [math.inf] * (n_v + n_w + 1)
        cross = delta_u(u) ** (-s)

        for t in range(n_v + n_w + 1):
            lo = max(0, t - n_w)
            hi = min(n_v, t)
            for t_v in range(lo, hi + 1):
                t_w = t - t_v
                candidate = F_v[t_v] + F_w[t_w] + t_v * t_w * cross
                if candidate < F_u[t] - 1e-15:
                    F_u[t] = candidate
        table[u] = F_u
        return F_u

    visit(root)
    return table


def dp_with_choices(root: Node, s: float) -> Tuple[Dict[Node, List[float]], Dict[Tuple[Node, int], Tuple[int, int]]]:
    """Compute DP values and argmin split choices."""
    table: Dict[Node, List[float]] = {}
    choice: Dict[Tuple[Node, int], Tuple[int, int]] = {}

    def visit(u: Node) -> List[float]:
        if u.is_leaf:
            F_u = [0.0, 0.0]
            table[u] = F_u
            return F_u

        assert u.left is not None and u.right is not None
        F_v = visit(u.left)
        F_w = visit(u.right)
        n_v = len(F_v) - 1
        n_w = len(F_w) - 1
        F_u = [math.inf] * (n_v + n_w + 1)
        cross = delta_u(u) ** (-s)

        for t in range(n_v + n_w + 1):
            lo = max(0, t - n_w)
            hi = min(n_v, t)
            best_split = (0, t)
            for t_v in range(lo, hi + 1):
                t_w = t - t_v
                candidate = F_v[t_v] + F_w[t_w] + t_v * t_w * cross
                if candidate < F_u[t] - 1e-15:
                    F_u[t] = candidate
                    best_split = (t_v, t_w)
            choice[(u, t)] = best_split
        table[u] = F_u
        return F_u

    visit(root)
    return table, choice


def reconstruct_subset(root: Node, k: int, choice: Dict[Tuple[Node, int], Tuple[int, int]]) -> Tuple[int, ...]:
    if root.is_leaf:
        return () if k == 0 else (root.leaf,)  # type: ignore[return-value]
    assert root.left is not None and root.right is not None
    k_left, k_right = choice[(root, k)]
    return tuple(sorted(reconstruct_subset(root.left, k_left, choice) + reconstruct_subset(root.right, k_right, choice)))


# ---------------------------------------------------------------------------
# Deterministic random-instance generator
# ---------------------------------------------------------------------------

def random_binary_ultrametric(n: int, seed: int) -> Node:
    """Generate a deterministic rooted binary ultrametric tree.

    Heights increase monotonically as clusters are merged.
    """
    rng = random.Random(seed)
    forest: List[Node] = [make_leaf(i) for i in range(n)]
    current_height = 0.2
    while len(forest) > 1:
        i, j = rng.sample(range(len(forest)), 2)
        if i > j:
            i, j = j, i
        a = forest.pop(j)
        b = forest.pop(i)
        current_height += rng.uniform(0.4, 1.1)
        forest.append(make_internal(a, b, current_height))
    return forest[0]


# ---------------------------------------------------------------------------
# Named examples
# ---------------------------------------------------------------------------

def primate_example_tree() -> Node:
    """Primate ultrametric tree used in the paper.

    H and C merge at 5.5 MYA, then with G at 6.0 MYA, then with O at 13.0 MYA.
    Labels: 0=H, 1=C, 2=G, 3=O.
    """
    H = make_leaf(0)
    C = make_leaf(1)
    G = make_leaf(2)
    O = make_leaf(3)
    HC = make_internal(H, C, 5.5)
    HCG = make_internal(HC, G, 6.0)
    return make_internal(HCG, O, 13.0)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def newick_like(node: Node) -> str:
    if node.is_leaf:
        return str(node.leaf)
    assert node.left is not None and node.right is not None
    return f"({newick_like(node.left)},{newick_like(node.right)}):{node.height:.3f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _demo() -> None:
    parser = argparse.ArgumentParser(description="Ultrametric DP for minimum Riesz s-energy subset selection")
    parser.add_argument("--example", choices=["primate", "random"], default="primate")
    parser.add_argument("--n", type=int, default=8, help="leaf count for random example")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--s", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--bruteforce", action="store_true", help="also solve by enumeration")
    args = parser.parse_args()

    root = primate_example_tree() if args.example == "primate" else random_binary_ultrametric(args.n, args.seed)
    table, choice = dp_with_choices(root, args.s)
    F_root = table[root]
    if args.k > len(F_root) - 1:
        raise ValueError("k exceeds number of leaves")
    subset = reconstruct_subset(root, args.k, choice)
    print(json.dumps({
        "tree": newick_like(root),
        "s": args.s,
        "k": args.k,
        "dp_value": F_root[args.k],
        "dp_subset": subset,
    }, indent=2))

    if args.bruteforce:
        bf_val, bf_subset = brute_force_best(root, args.k, args.s)
        print(json.dumps({
            "bruteforce_value": bf_val,
            "bruteforce_subset": bf_subset,
            "abs_diff": abs(bf_val - F_root[args.k]),
        }, indent=2))


if __name__ == "__main__":
    _demo()
