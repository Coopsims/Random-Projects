import numpy as np
import torch
import matplotlib.pyplot as plt
import time

def calculate_julia_set(width=10000, height=10000, max_iterations=300, c=-0.7 + 0.27j, batch_size=1000):
    """
    Calculate the Julia set for a given complex parameter c using PyTorch acceleration.
    Supports both CUDA (NVIDIA GPUs) and MPS (Apple Silicon GPUs) backends.

    Args:
        width (int): Width of the output image. Default is 5000.
        height (int): Height of the output image. Default is 5000.
        max_iterations (int): Maximum number of iterations. Default is 300.
        c (complex): Complex parameter that defines the Julia set. Default is -0.7 + 0.27j.
        batch_size (int): Number of rows to process in each batch. Default is 1000.

    Returns:
        numpy.ndarray: 2D array containing the iteration counts for each point.
    """
    # Check if CUDA or MPS (Apple Silicon) is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Create a grid of complex numbers
    real_range = np.linspace(-1.5, 1.5, width)
    imag_range = np.linspace(-1.5, 1.5, height)

    # Initialize the result array on CPU
    result = np.zeros((height, width), dtype=np.int32)

    # Convert complex parameter to real and imaginary parts
    c_real = float(c.real)
    c_imag = float(c.imag)

    # Process the image in batches to minimize data transfer
    for batch_start in range(0, height, batch_size):
        batch_end = min(batch_start + batch_size, height)
        batch_height = batch_end - batch_start

        # Create meshgrid for this batch
        x = torch.tensor(real_range, dtype=torch.float32, device=device)
        y = torch.tensor(imag_range[batch_start:batch_end], dtype=torch.float32, device=device)
        xv, yv = torch.meshgrid(x, y, indexing='ij')

        # Initialize complex plane for this batch
        zr = xv.transpose(0, 1)  # Transpose to match the expected output shape
        zi = yv.transpose(0, 1)

        # Initialize iteration count tensor
        iterations = torch.zeros_like(zr, dtype=torch.int32, device=device)

        # Create a mask for points that are still being iterated
        mask = torch.ones_like(zr, dtype=torch.bool, device=device)

        # Perform Julia set iteration
        for i in range(max_iterations):
            # Update only points that haven't escaped yet
            zr_new = torch.where(mask, zr * zr - zi * zi + c_real, zr)
            zi = torch.where(mask, 2 * zr * zi + c_imag, zi)
            zr = zr_new

            # Check which points have escaped
            escaped = (zr * zr + zi * zi) > 4.0

            # Update the iteration count for newly escaped points
            iterations = torch.where(escaped & mask, i + 1, iterations)

            # Update the mask to exclude escaped points
            mask = mask & (~escaped)

            # If all points have escaped, break early
            if not mask.any():
                break

        # Set the iteration count to max_iterations for points that never escaped
        # This matches the behavior of the CPU implementation
        iterations = torch.where(mask, max_iterations, iterations)

        # Copy the batch result back to CPU
        result[batch_start:batch_end, :] = iterations.cpu().numpy()

    return result

def display_julia_set(julia_set, acceleration_type='GPU', cmap='hot'):
    """
    Display the Julia set using matplotlib.

    Args:
        julia_set (numpy.ndarray): 2D array containing the iteration counts.
        acceleration_type (str): Type of acceleration used (CUDA, MPS, or CPU). Default is 'GPU'.
        cmap (str): Colormap to use for visualization. Default is 'hot'.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(julia_set, cmap=cmap)
    plt.title(f'Julia Set (PyTorch {acceleration_type} Accelerated)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    """
    Calculate and display the Julia set with default parameters using PyTorch acceleration.
    Supports both CUDA (NVIDIA GPUs) and MPS (Apple Silicon GPUs) backends.
    """
    try:
        # Check if PyTorch CUDA or MPS (Apple Silicon) is available
        if torch.cuda.is_available():
            print(f"CUDA GPU is available: {torch.cuda.get_device_name(0)}")
            print("Using PyTorch CUDA acceleration...")

            print("Calculating Julia set using CUDA GPU...")
            start_time = time.time()
            julia_set = calculate_julia_set()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Calculation completed in {elapsed_time:.4f} seconds")
            print("Displaying Julia set...")
            display_julia_set(julia_set, acceleration_type='CUDA')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("Apple Silicon GPU (MPS) is available")
            print("Using PyTorch MPS acceleration...")

            print("Calculating Julia set using Apple Silicon GPU...")
            start_time = time.time()
            julia_set = calculate_julia_set()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Calculation completed in {elapsed_time:.4f} seconds")
            print("Displaying Julia set...")
            display_julia_set(julia_set, acceleration_type='MPS')
        else:
            raise RuntimeError("No GPU acceleration available (neither CUDA nor MPS)")
    except (ImportError, RuntimeError) as e:
        print(f"PyTorch GPU acceleration not available: {e}")
        print("Falling back to CPU implementation...")

        # Import and use the CPU implementation
        from singlethread import calculate_julia_set as cpu_calculate_julia_set

        start_time = time.time()
        julia_set = cpu_calculate_julia_set()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"CPU calculation completed in {elapsed_time:.4f} seconds")
        display_julia_set(julia_set, acceleration_type='CPU')

if __name__ == "__main__":
    main()
