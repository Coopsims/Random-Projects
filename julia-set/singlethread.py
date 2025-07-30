import numpy as np
import matplotlib.pyplot as plt
import time

def calculate_julia_set(width=1000, height=1000, max_iterations=300, c=-0.7 + 0.27j):
    """
    Calculate the Julia set for a given complex parameter c.

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

    # Calculate the Julia set
    for y in range(height):
        for x in range(width):
            z = complex(real_range[x], imag_range[y])
            iteration = 0

            # Iterate until the point escapes or we reach max iterations
            while abs(z) <= 2 and iteration < max_iterations:
                z = z**2 + c
                iteration += 1

            result[y, x] = iteration

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
    plt.title('Julia Set')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    """
    Calculate and display the Julia set with default parameters.
    """
    print("Calculating Julia set...")
    start_time = time.time()
    julia_set = calculate_julia_set()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Calculation completed in {elapsed_time:.4f} seconds")
    print("Displaying Julia set...")
    display_julia_set(julia_set)

if __name__ == "__main__":
    main()
