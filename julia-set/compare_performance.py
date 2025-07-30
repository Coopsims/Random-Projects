import time
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib.util
import sys

def run_comparison(width=1000, height=1000, max_iterations=300, c=-0.7 + 0.27j, show_plots=False):
    """
    Run all Julia set implementations and compare their performance.

    Args:
        width (int): Width of the output image. Default is 1000.
        height (int): Height of the output image. Default is 1000.
        max_iterations (int): Maximum number of iterations. Default is 300.
        c (complex): Complex parameter that defines the Julia set. Default is -0.7 + 0.27j.
        show_plots (bool): Whether to display the Julia set plots. Default is False.
    """
    results = {}

    # Run single-threaded implementation
    print("\n=== Running Single-threaded CPU Implementation ===")
    from singlethread import calculate_julia_set as st_calculate_julia_set
    from singlethread import display_julia_set as st_display_julia_set

    start_time = time.time()
    st_result = st_calculate_julia_set(width, height, max_iterations, c)
    end_time = time.time()
    st_time = end_time - start_time
    results['Single-threaded CPU'] = st_time
    print(f"Single-threaded CPU: {st_time:.4f} seconds")

    if show_plots:
        st_display_julia_set(st_result)

    # Run multi-threaded implementation
    print("\n=== Running Multi-threaded CPU Implementation ===")
    try:
        # Use importlib to dynamically import the module with a hyphen in its name
        module_path = os.path.join(os.path.dirname(__file__), "multiprocessing-test.py")

        # Check if the file exists
        if not os.path.exists(module_path):
            raise ImportError(f"File not found: {module_path}")

        # Use the module name with a hyphen, which is the same as the file name without the .py extension
        module_name = "multiprocessing-test"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        multiprocessing_test = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = multiprocessing_test
        spec.loader.exec_module(multiprocessing_test)

        # Get the functions from the module
        mt_calculate_julia_set = multiprocessing_test.calculate_julia_set
        mt_display_julia_set = multiprocessing_test.display_julia_set

        start_time = time.time()
        mt_result = mt_calculate_julia_set(width, height, max_iterations, c)
        end_time = time.time()
        mt_time = end_time - start_time
        results['Multi-threaded CPU'] = mt_time
        speedup = st_time / mt_time
        print(f"Multi-threaded CPU: {mt_time:.4f} seconds (Speedup: {speedup:.2f}x)")

        if show_plots:
            mt_display_julia_set(mt_result)
    except ImportError:
        print("Multi-threaded implementation not available.")
    except Exception as e:
        print(f"Error running multi-threaded implementation: {e}")

    # Run PyTorch GPU implementation
    print("\n=== Running PyTorch GPU Implementation ===")
    try:
        import torch
        from pytorch_gpu import calculate_julia_set as pt_calculate_julia_set
        from pytorch_gpu import display_julia_set as pt_display_julia_set

        # Check if CUDA or MPS is available
        if torch.cuda.is_available():
            device_type = "CUDA"
            device_name = torch.cuda.get_device_name(0)
            print(f"Using {device_type} GPU: {device_name}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_type = "MPS"
            device_name = "Apple Silicon"
            print(f"Using {device_type} GPU: {device_name}")
        else:
            device_type = "CPU"
            device_name = "PyTorch CPU fallback"
            print("No GPU available, using PyTorch CPU fallback")

        start_time = time.time()
        pt_result = pt_calculate_julia_set(width, height, max_iterations, c)
        end_time = time.time()
        pt_time = end_time - start_time
        results[f'PyTorch {device_type}'] = pt_time
        speedup = st_time / pt_time
        print(f"PyTorch {device_type}: {pt_time:.4f} seconds (Speedup: {speedup:.2f}x)")

        if show_plots:
            pt_display_julia_set(pt_result, acceleration_type=device_type)
    except ImportError:
        print("PyTorch GPU implementation not available.")
    except Exception as e:
        print(f"Error running PyTorch GPU implementation: {e}")

    return results

def plot_results(results):
    """
    Plot the performance results as a bar chart.

    Args:
        results (dict): Dictionary mapping implementation names to execution times.
    """
    if not results:
        print("No results to plot.")
        return

    # Sort implementations by execution time (fastest first)
    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}

    # Calculate speedups relative to single-threaded CPU
    single_threaded_time = results.get('Single-threaded CPU', 1.0)
    speedups = {k: single_threaded_time / v for k, v in sorted_results.items()}

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot execution times
    plt.subplot(1, 2, 1)
    plt.bar(sorted_results.keys(), sorted_results.values(), color='skyblue')
    plt.title('Execution Time (lower is better)')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Plot speedups
    plt.subplot(1, 2, 2)
    plt.bar(speedups.keys(), speedups.values(), color='lightgreen')
    plt.title('Speedup vs. Single-threaded CPU (higher is better)')
    plt.ylabel('Speedup (x times)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.show()

def main():
    """
    Run the performance comparison with default parameters and plot the results.
    """
    print("=== Julia Set Performance Comparison ===")
    print("Comparing the performance of different Julia set implementations.")
    print("Parameters:")
    width = 7000
    height = 7000
    max_iterations = 300
    c = -0.7 + 0.27j
    print(f"  - Width: {width}")
    print(f"  - Height: {height}")
    print(f"  - Max iterations: {max_iterations}")
    print(f"  - Complex parameter c: {c}")

    # Run the comparison
    results = run_comparison(width, height, max_iterations, c, show_plots=False)

    # Print summary
    print("\n=== Performance Summary ===")
    if 'Single-threaded CPU' in results:
        baseline = results['Single-threaded CPU']
        print(f"Single-threaded CPU (baseline): {baseline:.4f} seconds")

        for impl, time_taken in results.items():
            if impl != 'Single-threaded CPU':
                speedup = baseline / time_taken
                print(f"{impl}: {time_taken:.4f} seconds (Speedup: {speedup:.2f}x)")
    else:
        print("Single-threaded CPU implementation not available for comparison.")

    # Plot the results
    plot_results(results)

if __name__ == "__main__":
    main()
