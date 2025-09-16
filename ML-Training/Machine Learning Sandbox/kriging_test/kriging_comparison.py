import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata, Rbf
from sklearn.linear_model import LinearRegression
import random
import argparse
import multiprocessing
from multiprocessing import Pool, cpu_count
import time
import os
from functools import partial

# Import functions from kriging.py
from kriging import (
    generate_random_points,
    calculate_variogram,
    fit_variogram_model,
    ordinary_kriging,
    ordinary_kriging_mp,
    inverse_distance_weighting,
    CO_WEST, CO_EAST, CO_NORTH, CO_SOUTH
)

# Try to import torch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def process_universal_kriging_point(args):
    """Worker function for multiprocessing version of universal kriging."""
    points, values, grid_point, variogram_params, trend_degree, i = args
    n = len(points)

    # Build trend matrix for data points
    if trend_degree == 1:
        # Linear trend: [1, x, y]
        F = np.column_stack((np.ones(n), points))
        p = 3  # Number of trend coefficients
    elif trend_degree == 2:
        # Quadratic trend: [1, x, y, x^2, xy, y^2]
        F = np.column_stack((
            np.ones(n),
            points,
            points[:, 0]**2,
            points[:, 0] * points[:, 1],
            points[:, 1]**2
        ))
        p = 6  # Number of trend coefficients
    else:
        raise ValueError("trend_degree must be 1 or 2")

    # Calculate distances from grid point to all data points
    distances = np.sqrt(np.sum((points - grid_point)**2, axis=1))

    # Build kriging matrix
    K = np.zeros((n+p, n+p))

    # Fill the variogram part of the matrix
    for j in range(n):
        for k in range(n):
            h = np.sqrt(np.sum((points[j] - points[k])**2))
            K[j, k] = exponential_variogram(h, 
                                          variogram_params["sill"], 
                                          variogram_params["range"], 
                                          variogram_params["nugget"])

    # Add trend components
    K[:n, n:] = F
    K[n:, :n] = F.T

    # Right-hand side: distances from prediction point to all data points
    k = np.zeros(n+p)
    for j in range(n):
        h = distances[j]
        k[j] = exponential_variogram(h, 
                                   variogram_params["sill"], 
                                   variogram_params["range"], 
                                   variogram_params["nugget"])

    # Build trend matrix for prediction point
    if trend_degree == 1:
        # Linear trend: [1, x, y]
        f_0 = np.array([1, grid_point[0], grid_point[1]])
    else:
        # Quadratic trend: [1, x, y, x^2, xy, y^2]
        f_0 = np.array([
            1, 
            grid_point[0], 
            grid_point[1], 
            grid_point[0]**2, 
            grid_point[0] * grid_point[1], 
            grid_point[1]**2
        ])

    k[n:] = f_0

    # Solve the kriging system
    try:
        weights = np.linalg.solve(K, k)
        # Calculate prediction
        prediction = np.sum(weights[:n] * values)
    except np.linalg.LinAlgError:
        # If matrix is singular, use inverse distance weighting as fallback
        if np.any(distances == 0):
            # If point coincides with a data point, use that value
            prediction = values[np.argmin(distances)]
        else:
            weights = 1.0 / distances
            weights = weights / np.sum(weights)
            prediction = np.sum(weights * values)

    return i, prediction

def universal_kriging(points, values, grid_points, variogram_params, trend_degree=1, n_jobs=None):
    """
    Perform universal kriging interpolation using multiprocessing.

    Universal kriging assumes a general polynomial trend model, unlike ordinary
    kriging which assumes a constant mean.

    Args:
        points: Array of shape (n, 2) with coordinates of data points
        values: Array of shape (n,) with values at data points
        grid_points: Array of shape (m, 2) with coordinates of grid points
        variogram_params: Dictionary with variogram parameters
        trend_degree: Degree of the polynomial trend (1=linear, 2=quadratic)
        n_jobs: Number of processes to use for multiprocessing (default: number of CPU cores)

    Returns:
        Array of shape (m,) with interpolated values at grid points
    """
    if n_jobs is None:
        n_jobs = cpu_count()

    # Create grid for prediction
    predictions = np.zeros(len(grid_points))

    # Prepare arguments for worker function
    args_list = [(points, values, grid_point, variogram_params, trend_degree, i) 
                for i, grid_point in enumerate(grid_points)]

    # Process grid points in parallel
    with Pool(processes=n_jobs) as pool:
        results = pool.map(process_universal_kriging_point, args_list)

    # Collect results
    for i, prediction in results:
        predictions[i] = prediction

    return predictions

def regression_kriging(points, values, grid_points, variogram_params, n_jobs=None):
    """
    Perform regression kriging interpolation.

    Regression kriging combines a regression model with ordinary kriging
    of the regression residuals.

    Args:
        points: Array of shape (n, 2) with coordinates of data points
        values: Array of shape (n,) with values at data points
        grid_points: Array of shape (m, 2) with coordinates of grid points
        variogram_params: Dictionary with variogram parameters
        n_jobs: Number of processes to use for multiprocessing (default: number of CPU cores)

    Returns:
        Array of shape (m,) with interpolated values at grid points
    """
    # Step 1: Fit a regression model to the data
    model = LinearRegression()
    model.fit(points, values)

    # Step 2: Calculate residuals
    trend = model.predict(points)
    residuals = values - trend

    # Step 3: Perform ordinary kriging on the residuals using multiprocessing
    residual_predictions = ordinary_kriging_mp(points, residuals, grid_points, variogram_params, n_jobs)

    # Step 4: Predict the trend at grid points
    trend_predictions = model.predict(grid_points)

    # Step 5: Combine trend and residual predictions
    predictions = trend_predictions + residual_predictions

    return predictions

def indicator_kriging(points, values, grid_points, variogram_params, threshold=None, n_jobs=None):
    """
    Perform indicator kriging interpolation.

    Indicator kriging estimates the probability of exceeding a threshold.

    Args:
        points: Array of shape (n, 2) with coordinates of data points
        values: Array of shape (n,) with values at data points
        grid_points: Array of shape (m, 2) with coordinates of grid points
        variogram_params: Dictionary with variogram parameters
        threshold: Threshold value (if None, use the median)
        n_jobs: Number of processes to use for multiprocessing (default: number of CPU cores)

    Returns:
        Array of shape (m,) with probability estimates at grid points
    """
    if threshold is None:
        threshold = np.median(values)

    # Convert values to indicators (0 or 1)
    indicators = (values > threshold).astype(float)

    # Perform ordinary kriging on the indicators using multiprocessing
    probabilities = ordinary_kriging_mp(points, indicators, grid_points, variogram_params, n_jobs)

    # Ensure probabilities are between 0 and 1
    probabilities = np.clip(probabilities, 0, 1)

    return probabilities

def radial_basis_function(points, values, grid_points, function='multiquadric', epsilon=1):
    """
    Perform Radial Basis Function interpolation.

    Args:
        points: Array of shape (n, 2) with coordinates of data points
        values: Array of shape (n,) with values at data points
        grid_points: Array of shape (m, 2) with coordinates of grid points
        function: RBF function type ('multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate')
        epsilon: Shape parameter for RBF

    Returns:
        Array of shape (m,) with interpolated values at grid points
    """
    # Create RBF interpolator
    rbf = Rbf(points[:, 0], points[:, 1], values, function=function, epsilon=epsilon)

    # Predict at grid points
    predictions = rbf(grid_points[:, 0], grid_points[:, 1])

    return predictions

def exponential_variogram(h, sill=1.0, range_param=1.0, nugget=0.0):
    """Exponential variogram model."""
    return nugget + sill * (1 - np.exp(-h / range_param))

def spherical_variogram(h, sill=1.0, range_param=1.0, nugget=0.0):
    """Spherical variogram model."""
    h = np.asarray(h)
    result = np.zeros_like(h, dtype=float)

    # h <= range_param
    mask = h <= range_param
    result[mask] = nugget + sill * (1.5 * (h[mask] / range_param) - 0.5 * (h[mask] / range_param)**3)

    # h > range_param
    mask = h > range_param
    result[mask] = nugget + sill

    return result

def gaussian_variogram(h, sill=1.0, range_param=1.0, nugget=0.0):
    """Gaussian variogram model."""
    return nugget + sill * (1 - np.exp(-(h**2) / (range_param**2)))

def run_task(task):
    """Worker function for multiprocessing that unpacks task arguments and calls run_interpolation_method."""
    args, kwargs = task
    return run_interpolation_method(*args, **kwargs)

def run_interpolation_method(method_name, points, values, grid_points, variogram_params=None, n_jobs=None, **kwargs):
    """
    Run a specific interpolation method and time its execution.

    Args:
        method_name: Name of the interpolation method
        points: Array of shape (n, 2) with coordinates of data points
        values: Array of shape (n,) with values at data points
        grid_points: Array of shape (m, 2) with coordinates of grid points
        variogram_params: Dictionary with variogram parameters (for kriging methods)
        n_jobs: Number of processes to use for multiprocessing (default: number of CPU cores)
        **kwargs: Additional parameters for specific methods

    Returns:
        Dictionary with method name, execution time, and interpolated values
    """
    start_time = time.time()

    try:
        if method_name == "ordinary_kriging":
            # Always use multiprocessing for ordinary kriging
            result = ordinary_kriging_mp(points, values, grid_points, variogram_params, n_jobs)
        elif method_name == "universal_kriging":
            trend_degree = kwargs.get("trend_degree", 1)
            # Use multiprocessing for universal kriging
            result = universal_kriging(points, values, grid_points, variogram_params, trend_degree, n_jobs)
        elif method_name == "regression_kriging":
            # Regression kriging uses ordinary_kriging_mp internally
            result = regression_kriging(points, values, grid_points, variogram_params, n_jobs)
        elif method_name == "indicator_kriging":
            threshold = kwargs.get("threshold", None)
            # Indicator kriging uses ordinary_kriging_mp internally
            result = indicator_kriging(points, values, grid_points, variogram_params, threshold, n_jobs)
        elif method_name == "idw":
            power = kwargs.get("power", 2.0)
            result = inverse_distance_weighting(points, values, grid_points, power)
        elif method_name == "rbf":
            function = kwargs.get("function", "multiquadric")
            epsilon = kwargs.get("epsilon", 1)
            result = radial_basis_function(points, values, grid_points, function, epsilon)
        else:
            raise ValueError(f"Unknown method: {method_name}")
    except Exception as e:
        print(f"Error in {method_name}: {e}")
        # Return griddata as fallback
        result = griddata(points, values, grid_points, method='cubic')

    # Fill NaN values with the mean of non-NaN values to ensure complete coverage
    if np.any(np.isnan(result)):
        print(f"Warning: NaN values detected in {method_name} results. Filling with mean value.")
        non_nan_mean = np.nanmean(result)
        result = np.nan_to_num(result, nan=non_nan_mean)

    end_time = time.time()
    execution_time = end_time - start_time

    return {
        "method": method_name,
        "time": execution_time,
        "result": result
    }

def compare_interpolation_methods(points, values, grid_points, grid_shape, methods_config, n_jobs=None):
    """
    Compare multiple interpolation methods using multiprocessing.

    Args:
        points: Array of shape (n, 2) with coordinates of data points
        values: Array of shape (n,) with values at data points
        grid_points: Array of shape (m, 2) with coordinates of grid points
        grid_shape: Tuple with shape of the grid (rows, cols)
        methods_config: List of dictionaries with method configurations
        n_jobs: Number of processes to use (default: number of CPU cores)

    Returns:
        List of dictionaries with results for each method
    """
    if n_jobs is None:
        n_jobs = cpu_count()

    print(f"Using {n_jobs} processes for parallel execution")

    # Calculate experimental variogram once for all kriging methods
    bin_centers, gamma = calculate_variogram(points, values)
    variogram_params = fit_variogram_model(bin_centers, gamma)
    print(f"Variogram parameters: {variogram_params}")

    # Prepare arguments for each method
    tasks = []
    for config in methods_config:
        method_name = config["method"]
        kwargs = config.get("params", {})

        # Add variogram parameters for kriging methods
        if "kriging" in method_name:
            task_args = (method_name, points, values, grid_points)
            task_kwargs = {"variogram_params": variogram_params, "n_jobs": n_jobs, **kwargs}
        else:
            task_args = (method_name, points, values, grid_points)
            task_kwargs = kwargs

        tasks.append((task_args, task_kwargs))

    # Run methods in parallel
    results = []

    # Execute all functions in parallel using the top-level run_task function
    with Pool(processes=n_jobs) as pool:
        results = pool.map(run_task, tasks)

    # Reshape results to match the grid
    for result in results:
        result["result"] = result["result"].reshape(grid_shape)

    return results

def plot_comparison(longitudes, latitudes, temperatures, grid_lon, grid_lat, results, output_dir=None):
    """
    Plot comparison of interpolation methods.

    Args:
        longitudes: Array of longitudes for data points
        latitudes: Array of latitudes for data points
        temperatures: Array of temperature values for data points
        grid_lon: 2D array of longitudes for grid points
        grid_lat: 2D array of latitudes for grid points
        results: List of dictionaries with results for each method
        output_dir: Directory to save output images (default: None)
    """
    # Calculate the aspect ratio for Colorado
    lon_range = CO_EAST - CO_WEST
    lat_range = CO_NORTH - CO_SOUTH
    aspect_ratio = np.cos(np.radians((CO_NORTH + CO_SOUTH) / 2))

    # Determine number of rows and columns for subplots
    # Add 1 to n_methods to account for the original data plot
    n_plots = len(results) + 1
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    # Create figure
    fig = plt.figure(figsize=(6*n_cols, 5*n_rows))

    # Plot original data points in the first subplot
    ax = fig.add_subplot(n_rows, n_cols, 1)
    scatter = ax.scatter(longitudes, latitudes, c=temperatures, cmap='coolwarm', 
                       s=50, edgecolor='k', norm=Normalize(vmin=30, vmax=90))
    plt.colorbar(scatter, ax=ax, label='Temperature (°F)')
    ax.set_title(f'Original Data Points (n={len(longitudes)})')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(CO_WEST, CO_EAST)
    ax.set_ylim(CO_SOUTH, CO_NORTH)
    ax.grid(True)
    ax.set_aspect(aspect_ratio)

    # Plot each interpolation method
    for i, result in enumerate(results):
        method_name = result["method"]
        execution_time = result["time"]
        interpolated_temps = result["result"]

        # Format method name for display
        display_name = method_name.replace('_', ' ').title()

        # Create subplot
        ax = fig.add_subplot(n_rows, n_cols, i+2)

        # Plot interpolated surface
        contour = ax.contourf(grid_lon, grid_lat, interpolated_temps, cmap='coolwarm', 
                            levels=20, norm=Normalize(vmin=30, vmax=90))
        plt.colorbar(contour, ax=ax, label='Temperature (°F)')

        # Add original data points
        ax.scatter(longitudes, latitudes, c='k', s=10, alpha=0.5)

        # Set title and labels
        ax.set_title(f'{display_name}\nTime: {execution_time:.2f}s')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xlim(CO_WEST, CO_EAST)
        ax.set_ylim(CO_SOUTH, CO_NORTH)
        ax.grid(True)
        ax.set_aspect(aspect_ratio)

    plt.tight_layout()

    # Save figure if output directory is specified
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, 'interpolation_methods_comparison.png')
        plt.savefig(filename, dpi=300)
        print(f"Comparison plot saved to '{filename}'")

    plt.show()

def main():
    """Main function to run the interpolation methods comparison."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare different interpolation methods using multiprocessing')
    parser.add_argument('--points', type=int, default=100,
                        help='Number of random data points to generate (default: 100)')
    parser.add_argument('--grid-size', type=int, default=50,
                        help='Size of the interpolation grid (default: 50x50)')
    parser.add_argument('--processes', type=int, default=12,
                        help='Number of processes to use (default: number of CPU cores)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output images (default: current directory)')
    args = parser.parse_args()

    # Set random seed for reproducibility
    print("Setting random seed to 42 for reproducibility...")
    np.random.seed(42)
    random.seed(42)
    if TORCH_AVAILABLE:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

    # Generate random temperature data points across Colorado
    print(f"Generating random temperature data points across Colorado (n={args.points})...")
    longitudes, latitudes, temperatures = generate_random_points(args.points)

    # Create a grid for interpolation
    print(f"Creating grid for interpolation ({args.grid_size}x{args.grid_size})...")
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(CO_WEST, CO_EAST, args.grid_size),
        np.linspace(CO_SOUTH, CO_NORTH, args.grid_size)
    )

    # Prepare data for interpolation
    points = np.column_stack((longitudes, latitudes))
    grid_points = np.column_stack((grid_lon.flatten(), grid_lat.flatten()))

    # Define methods to compare
    methods_config = [
        {"method": "ordinary_kriging"},
        {"method": "universal_kriging", "params": {"trend_degree": 1}},
        {"method": "regression_kriging"},
        {"method": "indicator_kriging", "params": {"threshold": 60}},
        {"method": "idw", "params": {"power": 2.0}},
        {"method": "rbf", "params": {"function": "multiquadric", "epsilon": 1}}
    ]

    # Compare interpolation methods
    print("Comparing interpolation methods...")
    results = compare_interpolation_methods(
        points, temperatures, grid_points, grid_lon.shape,
        methods_config, args.processes
    )

    # Sort results by execution time
    results.sort(key=lambda x: x["time"])

    # Print execution times
    print("\nExecution times:")
    for result in results:
        method_name = result["method"].replace('_', ' ').title()
        execution_time = result["time"]
        print(f"{method_name}: {execution_time:.2f} seconds")

    # Plot comparison
    print("\nPlotting comparison...")
    plot_comparison(
        longitudes, latitudes, temperatures,
        grid_lon, grid_lat, results,
        args.output_dir
    )

    print("\nComparison completed!")

if __name__ == "__main__":
    main()
