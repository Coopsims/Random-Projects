import os
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool, cpu_count


def sparse_matmul_worker(args):
    A_chunk, B, index = args
    # Uncomment the line below if you want to see the PID
    # print(f"Chunk {index} handled by process ID: {os.getpid()}")
    return A_chunk.dot(B)


def parallel_sparse_matmul(A, B, n_jobs=None):
    """
    Multiply two sparse matrices A and B in parallel.
    If n_jobs is not provided, defaults to the number of CPU cores.
    """
    if n_jobs is None:
        n_jobs = cpu_count()
        print(n_jobs)

    # Split A into row-based chunks
    n_rows = A.shape[0]
    chunk_size = (n_rows + n_jobs - 1) // n_jobs
    chunks = []
    for i in range(n_jobs):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_rows)
        if start >= n_rows:
            break
        # Pass index i for debugging (optional)
        chunks.append((A[start:end], B, i))

    # Create a pool of workers and map the chunks
    with Pool(processes=n_jobs) as pool:
        result_chunks = pool.map(sparse_matmul_worker, chunks)

    # Stack the results into a single sparse matrix
    return sp.vstack(result_chunks)


if __name__ == "__main__":
    from scipy.sparse import random as sparse_random

    # Adjust these dimensions (and density) if they're too large for your machine
    np.random.seed(0)
    m, k, n = 5000, 10000, 10000
    density = 0.5

    A = sparse_random(m, k, density=density, format='csr', random_state=0)
    B = sparse_random(k, n, density=density, format='csr', random_state=1)

    # Parallel multiplication with n_jobs determined by the number of CPU cores
    C = parallel_sparse_matmul(A, B)
    print("Result shape:", C.shape)
