"""
algo_bench.py — Minimal framework to benchmark algorithms and plot runtime graphs.

Usage:
    python algo_bench.py
Produces:
    - A linear runtime plot: runtime_vs_n.png
    - A log–log runtime plot: runtime_loglog.png
"""

from __future__ import annotations
import time
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class AlgoSpec:
    """Container for a single algorithm benchmark spec."""
    name: str
    func: Callable[..., Any]
    input_builder: Callable[[int, int], Tuple[tuple, dict]]
    sizes: List[int]


class BenchmarkRunner:
    """
    Lightweight benchmarking runner.

    Methods
    -------
    add_algorithm(name, func, input_builder, sizes)
        Register an algorithm with an input builder and the n values to test.
    run(repeats=5, warmup=1, rng_seed=42)
        Execute timing loops and collect results.
    plot_linear(save_path='runtime_vs_n.png')
        Plot runtime vs n on linear axes (one figure).
    plot_loglog(save_path='runtime_loglog.png')
        Plot runtime vs n on log–log axes (one figure).
    """
    def __init__(self):
        self._algos: List[AlgoSpec] = []
        self._results: Dict[str, List[Tuple[int, float]]] = {}

    def add_algorithm(
        self,
        name: str,
        func: Callable[..., Any],
        input_builder: Callable[[int, int], Tuple[tuple, dict]],
        sizes: Iterable[int],
    ) -> None:
        self._algos.append(AlgoSpec(name=name, func=func, input_builder=input_builder, sizes=list(sizes)))

    @property
    def results(self) -> Dict[str, List[Tuple[int, float]]]:
        return self._results

    @staticmethod
    def _time_once(fn: Callable[[], Any]) -> float:
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        return end - start

    def run(self, repeats: int = 5, warmup: int = 1, rng_seed: int = 42) -> Dict[str, List[Tuple[int, float]]]:
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        self._results = {}

        for spec in self._algos:
            series: List[Tuple[int, float]] = []
            for n in spec.sizes:
                # Build inputs; allow builder to use seed to ensure comparability
                args, kwargs = spec.input_builder(n, rng_seed)

                # Warmup calls
                for _ in range(warmup):
                    spec.func(*args, **kwargs)

                # Repeat timings; take the best to reduce noise
                times = []
                for _ in range(repeats):
                    t = self._time_once(lambda: spec.func(*args, **kwargs))
                    times.append(t)
                best = min(times)
                series.append((n, best))
            self._results[spec.name] = series
        return self._results

    def plot_linear(self, save_path: str = "runtime_vs_n.png") -> None:
        plt.figure()
        for name, series in self._results.items():
            xs = [n for n, _ in series]
            ys = [t for _, t in series]
            plt.plot(xs, ys, marker="o", label=name)
        plt.xlabel("Input size n")
        plt.ylabel("Runtime (s)")
        plt.title("Algorithm runtime vs n")
        plt.legend()
        plt.grid(True, which="both")
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        plt.close()

    def plot_loglog(self, save_path: str = "runtime_loglog.png") -> None:
        plt.figure()
        for name, series in self._results.items():
            xs = np.array([n for n, _ in series], dtype=float)
            ys = np.array([t for _, t in series], dtype=float)
            plt.loglog(xs, ys, marker="o", basex=10, basey=10, label=name)
            # Optional: estimate slope (complexity exponent) with linear fit in log space
            if len(xs) >= 2 and np.all(ys > 0):
                coeffs = np.polyfit(np.log10(xs), np.log10(ys), 1)  # slope, intercept
                slope = coeffs[0]
                print(f"{name}: estimated exponent ~ {slope:.3f}")
        plt.xlabel("log10(n)")
        plt.ylabel("log10(time in s)")
        plt.title("Algorithm runtime (log–log)")
        plt.legend()
        plt.grid(True, which="both")
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        plt.close()


# -------------------------
# Example algorithms & inputs
# -------------------------

def build_random_list(n: int, seed: int, low: int = 0, high: int = 10**6) -> Tuple[tuple, dict]:
    """
    Build a random list of integers.

    Parameters
    ----------
    n : int
        Input size.
    seed : int
        Seed used for reproducible builds.
    low, high : int
        Range of random integers.

    Returns
    -------
    args : tuple
        Positional args for the algorithm.
    kwargs : dict
        Keyword args for the algorithm.
    """
    rng = random.Random(seed + n)  # vary by n to avoid identical arrays
    arr = [rng.randint(low, high) for _ in range(n)]
    return (arr,), {}


def build_random_list_and_target(n: int, seed: int) -> Tuple[tuple, dict]:
    rng = random.Random(seed + 13 + n)
    arr = [rng.randint(0, 10**6) for _ in range(n)]
    target = arr[-1] if arr else 0  # worst-case for linear search
    return (arr, target), {}


def build_sorted_list_and_target(n: int, seed: int) -> Tuple[tuple, dict]:
    rng = random.Random(seed + 29 + n)
    arr = sorted(rng.randint(0, 10**6) for _ in range(n))
    target = arr[-1] if arr else 0
    return (arr, target), {}


# ---- Algorithms to compare ----

def sum_loop(arr: List[int]) -> int:
    """O(n) — sum via Python loop (avoids built-in sum speedups for clearer scaling)."""
    total = 0
    for x in arr:
        total += x
    return total


def timsort_sorted(arr: List[int]) -> List[int]:
    """O(n log n) average — Python's Timsort."""
    return sorted(arr)


def linear_search(arr: List[int], target: int) -> int:
    """O(n) — return index of target or -1."""
    for i, x in enumerate(arr):
        if x == target:
            return i
    return -1


def binary_search(arr: List[int], target: int) -> int:
    """O(log n) — binary search on a sorted list; return index or -1."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        elif arr[mid] > target:
            hi = mid - 1
        else:
            return mid
    return -1


def pair_count_quadratic(arr: List[int]) -> int:
    """
    O(n^2) — simple nested loop: count pairs (i<j) where arr[i] < arr[j].
    Intentionally quadratic but fast enough for small n.
    """
    cnt = 0
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            if ai < arr[j]:
                cnt += 1
    return cnt


# -------------------------
# Main: define sizes, run, plot
# -------------------------

def main() -> None:
    runner = BenchmarkRunner()

    # Choose n ranges per algorithm (keep quadratic small to avoid long runs)
    sizes_linear = [2**k for k in range(10, 17)]       # 1,024 .. 65,536
    sizes_nlogn  = [2**k for k in range(10, 17)]       # same as linear
    sizes_log    = [2**k for k in range(10, 20)]       # binary search can go larger
    sizes_quad   = [200, 400, 800, 1200, 1600]         # quadratic kept modest

    # Register algorithms
    runner.add_algorithm("Sum (O(n))", sum_loop, build_random_list, sizes_linear)
    runner.add_algorithm("Sort (O(n log n))", timsort_sorted, build_random_list, sizes_nlogn)
    runner.add_algorithm("Linear search (O(n))", linear_search, build_random_list_and_target, sizes_linear)
    runner.add_algorithm("Binary search (O(log n))", binary_search, build_sorted_list_and_target, sizes_log)
    runner.add_algorithm("Pair count (O(n^2))", pair_count_quadratic, build_random_list, sizes_quad)

    # Run benchmarks
    results = runner.run(repeats=5, warmup=1, rng_seed=42)

    # Print a compact table to stdout
    print("\nBest runtimes (seconds):")
    for name, series in results.items():
        row = ", ".join(f"n={n}: {t:.6f}s" for n, t in series)
        print(f"- {name}: {row}")

    # Make plots
    runner.plot_linear("runtime_vs_n.png")
    runner.plot_loglog("runtime_loglog.png")
    print("\nSaved plots: runtime_vs_n.png, runtime_loglog.png")

if __name__ == "__main__":
    main()