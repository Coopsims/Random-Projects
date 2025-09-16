import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import os
from functools import partial

def calculate_julia_row(y, width, real_range, imag_range, max_iterations, c):
    """
    Calculate one row of the Julia set.

    Args:
        y (int): The y-coordinate (row) to calculate.
        width (int): Width of the output image.
        real_range (numpy.ndarray): Array of real values for the complex grid.
        imag_range (numpy.ndarray): Array of imaginary values for the complex grid.
        max_iterations (int): Maximum number of iterations.
        c (complex): Complex parameter that defines the Julia set.

    Returns:
        tuple: A tuple containing the row index and the calculated row.
    """
    row = np.zeros(width, dtype=int)

    for x in range(width):
        z = complex(real_range[x], imag_range[y])
        iteration = 0

        # Iterate until the point escapes or we reach max iterations
        while abs(z) <= 2 and iteration < max_iterations:
            z = z**2 + c
            iteration += 1

        row[x] = iteration

    return y, row

def calculate_julia_set(width=2500, height=2500, max_iterations=300, c=-0.7 + 0.27j):
    """
    Calculate the Julia set for a given complex parameter c using multiprocessing.

    Args:
        width (int): Width of the output image. Default is 1000.
        height (int): Height of the output image. Default is 1000.
        max_iterations (int): Maximum number of iterations. Default is 300.
        c (complex): Complex parameter that defines the Julia set. Default is -0.7 + 0.27j.

    Returns:
        numpy.ndarray: 2D array containing the iteration counts for each point.
    """
    # Create a grid of complex numbers
    real_range = np.linspace(-1.5, 1.5, width)
    imag_range = np.linspace(-1.5, 1.5, height)

    # Initialize the result array
    result = np.zeros((height, width), dtype=int)

    # Get the number of CPU cores
    # Try different methods to get CPU count, fall back to default if none available
    num_cores = 4  # Default fallback value
    if hasattr(mp, 'cpu_count'):
        num_cores = mp.cpu_count()
    elif hasattr(os, 'cpu_count'):
        num_cores = os.cpu_count() or num_cores  # Use default if os.cpu_count() returns None
    # Create a pool of worker processes
    with mp.Pool(processes=num_cores) as pool:
        # Create a partial function with fixed parameters
        calculate_row = partial(
            calculate_julia_row,
            width=width,
            real_range=real_range,
            imag_range=imag_range,
            max_iterations=max_iterations,
            c=c
        )

        # Map the function to all rows and collect results
        for y, row in pool.map(calculate_row, range(height)):
            result[y] = row

    return result

def display_julia_set(julia_set, cmap='hot'):
    """
    Display the Julia set using matplotlib.

    Args:
        julia_set (numpy.ndarray): 2D array containing the iteration counts.
        cmap (str): Colormap to use for visualization. Default is 'hot'.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(julia_set, cmap=cmap)
    plt.title('Julia Set (Multiprocessing)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    """
    Calculate and display the Julia set with default parameters using multiprocessing.
    """
    print("Calculating Julia set using multiprocessing...")
    start_time = time.time()
    julia_set = calculate_julia_set()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Calculation completed in {elapsed_time:.4f} seconds")
    print("Displaying Julia set...")
    display_julia_set(julia_set)

if __name__ == "__main__":
    # This is required for Windows to avoid recursive spawning of processes
    # Check if freeze_support is available (Windows) and call it if it is
    if hasattr(mp, 'freeze_support'):
        mp.freeze_support()
    main()
