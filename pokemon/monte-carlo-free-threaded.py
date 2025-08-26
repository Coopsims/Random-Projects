#!/usr/bin/env python3
"""
Monte Carlo simulation for breeding perfect IVs, with multiprocessing and threading options.

This script ports the simulation logic from the notebook `monte-carlo-lucario.ipynb`
into a .py file and adds timing for:
  1) Multiprocessing using concurrent.futures.ProcessPoolExecutor
  2) Threading using concurrent.futures.ThreadPoolExecutor (best on Python 3.13t free-threaded)

Usage examples:
  python monte-carlo-free-threaded.py                    # defaults: trials=10_000_000, procs=cpu, threads=cpu
  python monte-carlo-free-threaded.py --trials 2000000 --processes 8 --threads 8
  python monte-carlo-free-threaded.py --same-nonbest     # both parents lack same non-best stat

Notes:
- The threaded run benefits significantly when using a free-threaded Python (3.13t) build.
- Both runs compute the same overall success rate over the same total number of trials.
"""
from __future__ import annotations
import os
import random
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Iterable, List, Sequence, Tuple


# --- Core chunk simulation (ported from the notebook) ---

def _simulate_chunk(args: Tuple[int, bool, int]) -> int:
    """Simulate a chunk of trials and return the number of successes.

    Args:
        args: (trials, diff_nonbest, seed)
    Returns:
        successes in this chunk
    """
    trials, diff_nonbest, seed = args
    # Use an independent RNG per worker to avoid shared-state contention
    rng = random.Random(seed)

    success = 0
    # Optimization notes:
    # - Instead of sampling 5-of-6 stats, pick the omitted stat uniformly (0..5).
    # - Only two stats (1 and 4) can be 0 in parents; others are always 31 when inherited.
    # - For diff_nonbest=True: P1 has 0 at stat 1, P2 has 0 at stat 4.
    #   Success requires: if stat 1 is inherited, choose P2; if stat 4 is inherited, choose P1.
    #   Then the omitted stat must roll 31 (1/32).
    # - For diff_nonbest=False: both parents have 0 at stat 1. Success requires omitted==1 and the roll is 31.
    if diff_nonbest:
        for _ in range(trials):
            omitted = rng.randrange(6)
            ok = True
            # If stat 1 is inherited, we must choose P2 (coin=1)
            if omitted != 1:
                ok = (rng.getrandbits(1) == 1)
            # If still ok and stat 4 is inherited, we must choose P1 (coin=0)
            if ok and omitted != 4:
                ok = (rng.getrandbits(1) == 0)
            if ok:
                # Omitted stat must roll 31
                if rng.getrandbits(5) == 31:
                    success += 1
    else:
        for _ in range(trials):
            # Must omit stat 1, and the random roll must be 31
            if rng.randrange(6) == 1 and rng.getrandbits(5) == 31:
                success += 1

    return success


def _split_trials_evenly(total_trials: int, workers: int) -> List[int]:
    per = total_trials // workers
    rem = total_trials % workers
    return [per + (1 if i < rem else 0) for i in range(workers)]


# --- Multiprocessing variant ---

def simulate_multiprocessing(
    trials: int = 10_000_000,
    diff_nonbest: bool = True,
    seed: int = 123,
    processes: int | None = None,
) -> float:
    """Run the simulation using multiple processes and return success rate.

    Mirrors the notebook's behavior with safe context fallback.
    """
    if processes is None:
        processes = os.cpu_count() or 1
    if processes <= 1:
        # Single-process fallback
        return _simulate_chunk((trials, diff_nonbest, seed)) / trials

    # Split trials across processes
    base_seed = (seed * 1000003) % (2**31 - 1)
    sizes = [t for t in _split_trials_evenly(trials, processes) if t > 0]
    tasks = [
        (t, diff_nonbest, base_seed + i + 1)
        for i, t in enumerate(sizes)
    ]

    successes = None
    # Prefer 'fork' when available to avoid notebook pickling issues; fall back gracefully.
    try:
        ctx = mp.get_context('fork')
        with ProcessPoolExecutor(max_workers=processes, mp_context=ctx) as executor:
            successes = list(executor.map(_simulate_chunk, tasks))
    except (AttributeError, ValueError, RuntimeError, BrokenProcessPool):
        try:
            with ProcessPoolExecutor(max_workers=processes) as executor:
                successes = list(executor.map(_simulate_chunk, tasks))
        except BrokenProcessPool:
            # As a last resort, run single-process
            return _simulate_chunk((trials, diff_nonbest, seed)) / trials

    total_success = sum(successes)
    return total_success / trials


# --- Threaded variant (benefits on free-threaded Python 3.13t) ---

def simulate_threaded(
    trials: int = 10_000_000,
    diff_nonbest: bool = True,
    seed: int = 123,
    threads: int | None = None,
) -> float:
    """Run the simulation using threads and return success rate.

    Each thread runs an independent chunk with its own seed. This approach
    benefits from Python 3.13t (free-threaded, no GIL). On standard CPython,
    it may not scale due to the GIL.
    """
    if threads is None:
        threads = os.cpu_count() or 1
    if threads <= 1:
        return _simulate_chunk((trials, diff_nonbest, seed)) / trials

    sizes = [t for t in _split_trials_evenly(trials, threads) if t > 0]
    base_seed = (seed * 1000003) % (2**31 - 1)
    seeds = [base_seed + i + 1 for i in range(len(sizes))]

    successes: List[int] = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(_simulate_chunk, (t, diff_nonbest, s))
            for t, s in zip(sizes, seeds)
        ]
        for fut in futures:
            successes.append(fut.result())

    total_success = sum(successes)
    return total_success / trials


# --- Benchmark orchestration and CLI ---

if __name__ == "__main__":
    # Fixed configuration (no CLI args)
    trials = 10_000_000
    diff_nonbest = True
    seed = 123
    processes = os.cpu_count() or 1
    threads = 16

    print(f"Python: {sys.version.splitlines()[0]}")
    print(f"Platform: {sys.platform}, CPU count: {os.cpu_count()}\n")

    # Multiprocessing run
    t0 = time.perf_counter()
    rate_mp = simulate_multiprocessing(trials=trials, diff_nonbest=diff_nonbest, seed=seed, processes=processes)
    dt_mp = time.perf_counter() - t0
    print(f"Multiprocessing: processes={processes}, trials={trials}")
    print(f"  Success rate: {rate_mp:.12f}")
    print(f"  Time: {dt_mp:.3f} s\n")

    # Threaded run (32 threads)
    t1 = time.perf_counter()
    rate_th = simulate_threaded(trials=trials, diff_nonbest=diff_nonbest, seed=seed, threads=threads)
    dt_th = time.perf_counter() - t1
    print(f"Threading: threads={threads}, trials={trials}")
    print(f"  Success rate: {rate_th:.12f}")
    print(f"  Time: {dt_th:.3f} s")
