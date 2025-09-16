import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import random
import argparse
import multiprocessing
from multiprocessing import Pool, cpu_count
import time

# Try to import torch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Define Colorado's approximate boundaries (longitude and latitude)
CO_WEST = -109.05
CO_EAST = -102.05
CO_NORTH = 41.0
CO_SOUTH = 37.0

def generate_random_points(n=100):
    """Generate n random points within Colorado's boundaries.

    Points follow a semi-normal distribution across longitude.
    Temperatures follow a warm-cool-warm pattern from west to east with some variation.
    """
    # Create a semi-normal distribution of points across longitude
    # Use a mixture of normal distributions centered at different points
    mix_ratio = 0.7
    normal_points = np.random.normal(loc=(CO_WEST + CO_EAST) / 2, scale=(CO_EAST - CO_WEST) / 6, size=int(n * mix_ratio))
    uniform_points = np.random.uniform(CO_WEST, CO_EAST, n - int(n * mix_ratio))
    longitudes = np.concatenate([normal_points, uniform_points])

    # Clip to ensure all points are within Colorado's boundaries
    longitudes = np.clip(longitudes, CO_WEST, CO_EAST)

    # Generate random latitudes (uniform distribution)
    latitudes = np.random.uniform(CO_SOUTH, CO_NORTH, n)

    # Generate temperatures with a warm-cool-warm pattern from west to east
    # Map longitude to a position in the pattern
    normalized_lon = (longitudes - CO_WEST) / (CO_EAST - CO_WEST)  # 0 to 1

    # Create a temperature pattern: warm at edges, cooler in middle
    # Use a cosine function to create the pattern
    base_temp = 60  # Base temperature
    temp_range = 40  # Temperature range

    # Create a pattern that's warm at the edges and cooler in the middle
    # The pattern is centered around longitude -107.0 (approximately middle of Colorado)
    cool_center = (-106.0 - CO_WEST) / (CO_EAST - CO_WEST)  # Normalize the cool center

    # Calculate distance from the cool center (0 to 1)
    dist_from_center = np.abs(normalized_lon - cool_center)

    # Create the temperature pattern
    pattern = base_temp + temp_range * (dist_from_center * 1.5)

    # Make the mountains (western part of Colorado) colder
    # Mountains are approximately west of longitude -105.5
    mountain_boundary = (-105.5 - CO_WEST) / (CO_EAST - CO_WEST)  # Normalize the mountain boundary
    mountain_mask = normalized_lon < mountain_boundary

    # Apply temperature reduction to mountain regions
    # The further west (more mountainous), the colder it gets
    mountain_factor = np.zeros(n)
    mountain_factor[mountain_mask] = (mountain_boundary - normalized_lon[mountain_mask]) * 25  # Adjust the multiplier for desired effect
    pattern = pattern - mountain_factor

    # Add random variation
    variation = np.random.normal(0, 15, n)
    temperatures = pattern + variation

    # Clip to reasonable temperature range
    temperatures = np.clip(temperatures, 20, 90)  # Lower minimum temperature to allow for colder mountains

    return longitudes, latitudes, temperatures

def exponential_variogram(h, sill=1.0, range_param=1.0, nugget=0.0):
    """Exponential variogram model."""
    return nugget + sill * (1 - np.exp(-h / range_param))

def calculate_variogram(points, values):
    """Calculate experimental variogram from data points."""
    n = len(points)
    distances = cdist(points, points)

    # Calculate semivariance
    h_bins = np.linspace(0, np.max(distances), 10)
    gamma = np.zeros(len(h_bins)-1)

    for i in range(len(h_bins)-1):
        mask = (distances > h_bins[i]) & (distances <= h_bins[i+1])
        if np.sum(mask) > 0:
            squared_diff = 0
            count = 0
            for j in range(n):
                for k in range(j+1, n):
                    if h_bins[i] < distances[j, k] <= h_bins[i+1]:
                        squared_diff += (values[j] - values[k])**2
                        count += 1
            if count > 0:
                gamma[i] = squared_diff / (2 * count)

    # Return bin centers and semivariance values
    bin_centers = (h_bins[:-1] + h_bins[1:]) / 2
    return bin_centers, gamma

def fit_variogram_model(bin_centers, gamma):
    """Fit an exponential variogram model to the experimental variogram."""
    # Simple estimation of parameters
    nugget = np.min(gamma) if len(gamma) > 0 and not np.isnan(gamma[0]) else 0
    sill = np.max(gamma) - nugget if len(gamma) > 0 else 1.0
    range_param = bin_centers[np.argmax(gamma)] / 3 if len(gamma) > 0 else 1.0

    return {"nugget": nugget, "sill": sill, "range": range_param}

def ordinary_kriging(points, values, grid_points, variogram_params):
    """Perform ordinary kriging interpolation (single-threaded CPU version)."""
    n = len(points)

    # Create grid for prediction
    predictions = np.zeros(len(grid_points))

    for i, grid_point in enumerate(grid_points):
        # Calculate distances from grid point to all data points
        distances = np.sqrt(np.sum((points - grid_point)**2, axis=1))

        # Build kriging matrix
        K = np.zeros((n+1, n+1))
        for j in range(n):
            for k in range(n):
                h = np.sqrt(np.sum((points[j] - points[k])**2))
                K[j, k] = exponential_variogram(h, 
                                              variogram_params["sill"], 
                                              variogram_params["range"], 
                                              variogram_params["nugget"])

        # Add lagrange multiplier constraints
        K[:n, n] = 1.0
        K[n, :n] = 1.0
        K[n, n] = 0.0

        # Right-hand side: distances from prediction point to all data points
        k = np.zeros(n+1)
        for j in range(n):
            h = distances[j]
            k[j] = exponential_variogram(h, 
                                       variogram_params["sill"], 
                                       variogram_params["range"], 
                                       variogram_params["nugget"])
        k[n] = 1.0

        # Solve the kriging system
        try:
            weights = np.linalg.solve(K, k)
            # Calculate prediction
            predictions[i] = np.sum(weights[:n] * values)
        except np.linalg.LinAlgError:
            # If matrix is singular, use inverse distance weighting as fallback
            if np.any(distances == 0):
                # If point coincides with a data point, use that value
                predictions[i] = values[np.argmin(distances)]
            else:
                weights = 1.0 / distances
                weights = weights / np.sum(weights)
                predictions[i] = np.sum(weights * values)

    return predictions

def process_grid_point(args):
    """Worker function for multiprocessing version of kriging."""
    points, values, grid_point, variogram_params, i = args
    n = len(points)

    # Calculate distances from grid point to all data points
    distances = np.sqrt(np.sum((points - grid_point)**2, axis=1))

    # Build kriging matrix
    K = np.zeros((n+1, n+1))
    for j in range(n):
        for k in range(n):
            h = np.sqrt(np.sum((points[j] - points[k])**2))
            K[j, k] = exponential_variogram(h, 
                                          variogram_params["sill"], 
                                          variogram_params["range"], 
                                          variogram_params["nugget"])

    # Add lagrange multiplier constraints
    K[:n, n] = 1.0
    K[n, :n] = 1.0
    K[n, n] = 0.0

    # Right-hand side: distances from prediction point to all data points
    k = np.zeros(n+1)
    for j in range(n):
        h = distances[j]
        k[j] = exponential_variogram(h, 
                                   variogram_params["sill"], 
                                   variogram_params["range"], 
                                   variogram_params["nugget"])
    k[n] = 1.0

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

def ordinary_kriging_mp(points, values, grid_points, variogram_params, n_jobs=None):
    """Perform ordinary kriging interpolation using multiprocessing."""
    if n_jobs is None:
        n_jobs = cpu_count()

    # Create grid for prediction
    predictions = np.zeros(len(grid_points))

    # Prepare arguments for worker function
    args_list = [(points, values, grid_point, variogram_params, i) 
                for i, grid_point in enumerate(grid_points)]

    # Process grid points in parallel
    with Pool(processes=n_jobs) as pool:
        results = pool.map(process_grid_point, args_list)

    # Collect results
    for i, prediction in results:
        predictions[i] = prediction

    return predictions

def exponential_variogram_torch(h, sill=1.0, range_param=1.0, nugget=0.0):
    """Exponential variogram model for PyTorch tensors."""
    return nugget + sill * (1 - torch.exp(-h / range_param))

def ordinary_kriging_gpu(points, values, grid_points, variogram_params):
    """Perform ordinary kriging interpolation using GPU acceleration with PyTorch.

    This implementation uses batched operations and vectorization to maximize GPU efficiency:
    - Processes grid points in batches
    - Uses vectorized operations instead of loops
    - Minimizes CPU-GPU data transfers
    - Uses batched linear algebra operations
    - Provides MPS-specific optimizations for Apple Silicon
    """
    if not TORCH_AVAILABLE:
        print("PyTorch is not available. Falling back to multiprocessing.")
        return ordinary_kriging_mp(points, values, grid_points, variogram_params)

    # Determine the device to use
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
        using_mps = False
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
        using_mps = True
    else:
        device = torch.device("cpu")
        print("No GPU available. Using CPU with PyTorch")
        using_mps = False

    # Convert numpy arrays to PyTorch tensors and move to device
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)

    n = len(points)
    m = len(grid_points)

    # Extract variogram parameters and convert to tensors
    sill = torch.tensor(variogram_params["sill"], dtype=torch.float32, device=device)
    range_param = torch.tensor(variogram_params["range"], dtype=torch.float32, device=device)
    nugget = torch.tensor(variogram_params["nugget"], dtype=torch.float32, device=device)

    # Pre-compute distances between all data points
    point_distances = torch.cdist(points_tensor, points_tensor)

    # Vectorized computation of the kriging matrix using the exponential variogram
    K_base = exponential_variogram_torch(point_distances, sill, range_param, nugget)

    # Add Lagrange multiplier constraints
    K = torch.zeros((n+1, n+1), dtype=torch.float32, device=device)
    K[:n, :n] = K_base
    K[:n, n] = 1.0
    K[n, :n] = 1.0

    # Compute distances from all grid points to all data points at once
    # This is a key optimization - computing all distances in a single operation
    grid_to_point_distances = torch.cdist(grid_points_tensor, points_tensor)

    # Process grid points in batches to avoid memory issues on large grids
    # Use smaller batch size for MPS to avoid memory issues
    batch_size = min(100 if using_mps else 1000, m)
    predictions = torch.zeros(m, dtype=torch.float32, device=device)

    # For MPS, precompute the LU factorization of K if possible
    # This avoids repeated calls to torch.linalg.solve which can be unstable on MPS
    if using_mps:
        try:
            # Add a small value to the diagonal for numerical stability
            K_stable = K.clone()
            K_stable[:n, :n] = K_stable[:n, :n] + torch.eye(n, device=device) * 1e-6

            # Try to use torch.linalg.lu_factor which is more stable than solve
            # If this fails, we'll fall back to the regular approach
            lu_factors = torch.linalg.lu_factor(K_stable)
            using_lu = True
            print("Using LU factorization for more stable solving on MPS")
        except (RuntimeError, AttributeError):
            using_lu = False
            print("LU factorization not available, using standard solver")
    else:
        using_lu = False

    for batch_start in range(0, m, batch_size):
        batch_end = min(batch_start + batch_size, m)
        batch_indices = torch.arange(batch_start, batch_end, device=device)
        batch_size_actual = len(batch_indices)

        # Get distances for this batch
        batch_distances = grid_to_point_distances[batch_indices]

        # Vectorized computation of the right-hand side vectors for all grid points in the batch
        # Shape: [batch_size, n]
        k_vectors = exponential_variogram_torch(batch_distances, sill, range_param, nugget)

        # Add Lagrange multiplier (1.0) to each vector
        # Shape: [batch_size, n+1]
        k_vectors_with_lagrange = torch.ones((batch_size_actual, n+1), dtype=torch.float32, device=device)
        k_vectors_with_lagrange[:, :n] = k_vectors

        # Handle special case: if any grid point coincides with a data point
        zero_distances = torch.isclose(batch_distances, torch.tensor(0.0, device=device))
        has_zero_distance = torch.any(zero_distances, dim=1)

        # For points with zero distances, we'll use the exact value
        if torch.any(has_zero_distance):
            for i in range(batch_size_actual):
                if has_zero_distance[i]:
                    # Find which data point(s) coincide with this grid point
                    zero_indices = torch.where(zero_distances[i])[0]
                    # Use the value of the first coinciding data point
                    predictions[batch_indices[i]] = values_tensor[zero_indices[0]]

        # For other points, solve the kriging system
        non_zero_indices = torch.where(~has_zero_distance)[0]
        if len(non_zero_indices) > 0:
            try:
                # Different solving approaches based on device and availability
                if using_lu:
                    # Use the precomputed LU factorization for more stable solving on MPS
                    batch_predictions = torch.zeros(len(non_zero_indices), dtype=torch.float32, device=device)

                    # Process each vector individually with lu_solve
                    for i, idx in enumerate(non_zero_indices):
                        weights = torch.linalg.lu_solve(
                            lu_factors,
                            k_vectors_with_lagrange[idx].unsqueeze(1)
                        ).squeeze(1)
                        batch_predictions[i] = torch.sum(weights[:n] * values_tensor)
                else:
                    # For CUDA or CPU, or if LU factorization is not available on MPS
                    # Process in smaller sub-batches if using MPS to improve stability
                    sub_batch_size = 10 if using_mps else len(non_zero_indices)
                    batch_predictions = torch.zeros(len(non_zero_indices), dtype=torch.float32, device=device)

                    for sub_start in range(0, len(non_zero_indices), sub_batch_size):
                        sub_end = min(sub_start + sub_batch_size, len(non_zero_indices))
                        sub_indices = non_zero_indices[sub_start:sub_end]

                        # Add a small value to the diagonal for numerical stability on MPS
                        if using_mps:
                            K_sub = K.clone()
                            K_sub[:n, :n] = K_sub[:n, :n] + torch.eye(n, device=device) * 1e-6
                            K_batch = K_sub.unsqueeze(0).expand(len(sub_indices), -1, -1)
                        else:
                            K_batch = K.unsqueeze(0).expand(len(sub_indices), -1, -1)

                        # Solve the system
                        weights = torch.linalg.solve(
                            K_batch,
                            k_vectors_with_lagrange[sub_indices]
                        )

                        # Calculate predictions
                        sub_predictions = torch.sum(weights[:, :n] * values_tensor.unsqueeze(0), dim=1)
                        batch_predictions[sub_start:sub_end] = sub_predictions

                # Store the results
                predictions[batch_indices[non_zero_indices]] = batch_predictions

            except RuntimeError as e:
                # More detailed error message for debugging
                if using_mps:
                    print(f"Linear solve failed on MPS: {e}")
                    print("Using IDW as fallback for this batch.")
                else:
                    print(f"Linear solve failed: {e}")
                    print("Using IDW as fallback for this batch.")

                # Fallback to IDW if linear solve fails
                for i in non_zero_indices:
                    distances = batch_distances[i]
                    if torch.any(torch.isclose(distances, torch.tensor(0.0, device=device))):
                        # If point coincides with a data point, use that value
                        predictions[batch_indices[i]] = values_tensor[torch.argmin(distances)]
                    else:
                        # Otherwise use inverse distance weighting
                        weights = 1.0 / distances
                        weights = weights / torch.sum(weights)
                        predictions[batch_indices[i]] = torch.sum(weights * values_tensor)

    # Move result back to CPU and convert to numpy
    return predictions.cpu().numpy()

def inverse_distance_weighting(points, values, grid_points, power=2):
    """
    Perform inverse distance weighting (IDW) interpolation.

    This is a simpler and faster alternative to kriging, but still provides
    reasonable results for spatial interpolation.

    Args:
        points: Array of shape (n, 2) with coordinates of data points
        values: Array of shape (n,) with values at data points
        grid_points: Array of shape (m, 2) with coordinates of grid points
        power: Power parameter for IDW (default: 2)

    Returns:
        Array of shape (m,) with interpolated values at grid points
    """
    # Calculate distances from all grid points to all data points
    distances = cdist(grid_points, points)

    # Handle zero distances (exact matches with data points)
    zero_dist_mask = distances == 0
    row_has_zero = np.any(zero_dist_mask, axis=1)

    # Initialize predictions
    predictions = np.zeros(len(grid_points))

    # For grid points that match data points exactly, use the data value
    for i in np.where(row_has_zero)[0]:
        data_idx = np.where(zero_dist_mask[i])[0][0]
        predictions[i] = values[data_idx]

    # For other grid points, use IDW
    non_zero_dist_mask = ~row_has_zero
    if np.any(non_zero_dist_mask):
        # Calculate weights as 1/distance^power
        weights = 1.0 / (distances[non_zero_dist_mask] ** power)

        # Normalize weights to sum to 1
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        normalized_weights = weights / weights_sum

        # Calculate predictions as weighted sum of values
        predictions[non_zero_dist_mask] = np.sum(normalized_weights * values, axis=1)

    return predictions

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Kriging interpolation with GPU or multiprocessing support')
    parser.add_argument('--mode', type=str, choices=['cpu', 'mp', 'gpu', 'idw'], default='cpu',
                        help='Processing mode: cpu (single-threaded), mp (multiprocessing), gpu, or idw (faster alternative)')
    parser.add_argument('--processes', type=int, default=None,
                        help='Number of processes to use for multiprocessing (default: number of CPU cores)')
    parser.add_argument('--points', type=int, default=100,
                        help='Number of random data points to generate (default: 100)')
    parser.add_argument('--grid-size', type=int, default=100,
                        help='Size of the interpolation grid (default: 50x50)')
    parser.add_argument('--idw-power', type=float, default=2.0,
                        help='Power parameter for IDW interpolation (default: 2.0)')
    return parser.parse_args()

def main(args=None):
    # Parse command line arguments if not provided
    if args is None:
        args = parse_arguments()

    # Set random seed for reproducibility
    print("Setting random seed to 42 for reproducibility...")
    np.random.seed(42)
    random.seed(42)
    if TORCH_AVAILABLE:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

    # Generate random temperature data points across Colorado
    print("Generating random temperature data points across Colorado...")
    longitudes, latitudes, temperatures = generate_random_points(args.points)

    # Create a grid for interpolation
    print(f"Creating grid for interpolation ({args.grid_size}x{args.grid_size})...")
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(CO_WEST, CO_EAST, args.grid_size),
        np.linspace(CO_SOUTH, CO_NORTH, args.grid_size)
    )

    # Prepare data for kriging
    points = np.column_stack((longitudes, latitudes))
    grid_points = np.column_stack((grid_lon.flatten(), grid_lat.flatten()))

    # Calculate experimental variogram
    print("Calculating experimental variogram...")
    bin_centers, gamma = calculate_variogram(points, temperatures)

    # Fit variogram model
    print("Fitting variogram model...")
    variogram_params = fit_variogram_model(bin_centers, gamma)
    print(f"Variogram parameters: {variogram_params}")

    # For simplicity, we'll use scipy's griddata for interpolation
    # This is not true kriging but a simpler interpolation method
    print("Performing interpolation...")
    grid_temp = griddata(points, temperatures, (grid_lon, grid_lat), method='cubic')

    # Fill NaN values in grid_temp with the mean of non-NaN values
    if np.any(np.isnan(grid_temp)):
        print("Warning: NaN values detected in griddata results. Filling with mean value.")
        non_nan_mean = np.nanmean(grid_temp)
        grid_temp = np.nan_to_num(grid_temp, nan=non_nan_mean)

    # Perform interpolation based on the selected mode
    if args.mode == 'idw':
        print(f"Performing Inverse Distance Weighting (IDW) interpolation (power={args.idw_power})...")
        method_name = "IDW"
    else:
        print(f"Performing kriging using {args.mode.upper()} mode (this may take a moment)...")
        method_name = f"Kriging ({args.mode.upper()})"

    start_time = time.time()

    try:
        if args.mode == 'idw':
            # Use the faster IDW method
            interpolated_temps = inverse_distance_weighting(points, temperatures, grid_points, power=args.idw_power)
        elif args.mode == 'gpu':
            if not TORCH_AVAILABLE:
                print("PyTorch is not available. Falling back to multiprocessing.")
                interpolated_temps = ordinary_kriging_mp(points, temperatures, grid_points, variogram_params, args.processes)
            else:
                interpolated_temps = ordinary_kriging_gpu(points, temperatures, grid_points, variogram_params)
        elif args.mode == 'mp':
            # Multiprocessing mode uses parallel processing
            interpolated_temps = ordinary_kriging_mp(points, temperatures, grid_points, variogram_params, args.processes)
        elif args.mode == 'cpu':
            # CPU mode uses single-core processing
            interpolated_temps = ordinary_kriging(points, temperatures, grid_points, variogram_params)

        # Fill NaN values with the mean of non-NaN values to ensure complete coverage
        if np.any(np.isnan(interpolated_temps)):
            print(f"Warning: NaN values detected in interpolation results. Filling with mean value.")
            non_nan_mean = np.nanmean(interpolated_temps)
            interpolated_temps = np.nan_to_num(interpolated_temps, nan=non_nan_mean)

        interpolated_temps = interpolated_temps.reshape(grid_lon.shape)
        use_interpolation = True

        end_time = time.time()
        print(f"Interpolation completed in {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error in interpolation: {e}")
        print("Falling back to griddata interpolation...")
        use_interpolation = False

    # Plot the results
    print("Plotting results...")
    plt.figure(figsize=(15, 10))

    # Calculate the aspect ratio for Colorado
    # Colorado is roughly rectangular, so we need to maintain the correct aspect ratio
    lon_range = CO_EAST - CO_WEST
    lat_range = CO_NORTH - CO_SOUTH
    # The aspect ratio should be the cosine of the mean latitude (to account for longitude compression)
    aspect_ratio = np.cos(np.radians((CO_NORTH + CO_SOUTH) / 2))

    # Plot 1: Original data points
    ax1 = plt.subplot(1, 2, 1)
    scatter = plt.scatter(longitudes, latitudes, c=temperatures, cmap='coolwarm', 
                         s=50, edgecolor='k', norm=Normalize(vmin=30, vmax=90))
    plt.colorbar(scatter, label='Temperature (°F)')
    plt.title(f'Original Random Temperature Data Points (n={args.points})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xlim(CO_WEST, CO_EAST)
    plt.ylim(CO_SOUTH, CO_NORTH)
    plt.grid(True)
    # Set aspect ratio to maintain the correct shape of Colorado
    ax1.set_aspect(aspect_ratio)

    # Plot 2: Interpolated surface
    ax2 = plt.subplot(1, 2, 2)
    if use_interpolation:
        contour = plt.contourf(grid_lon, grid_lat, interpolated_temps, cmap='coolwarm', 
                              levels=20, norm=Normalize(vmin=30, vmax=90))
        plt.title(f'{method_name} Interpolation')
    else:
        contour = plt.contourf(grid_lon, grid_lat, grid_temp, cmap='coolwarm', 
                              levels=20, norm=Normalize(vmin=30, vmax=90))
        plt.title('Cubic Interpolation (griddata)')

    plt.colorbar(contour, label='Temperature (°F)')
    plt.scatter(longitudes, latitudes, c='k', s=10, alpha=0.5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.xlim(CO_WEST, CO_EAST)
    plt.ylim(CO_SOUTH, CO_NORTH)
    plt.grid(True)
    # Set aspect ratio to maintain the correct shape of Colorado
    ax2.set_aspect(aspect_ratio)

    plt.tight_layout()

    # Create a filename that reflects the interpolation method used
    if args.mode == 'idw':
        filename = f'colorado_temperature_idw_power{args.idw_power}.png'
    else:
        filename = f'colorado_temperature_kriging_{args.mode}.png'

    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"Done! Results saved to '{filename}'")

def run_with_mode(mode, points=100, grid_size=50, processes=None, idw_power=2.0):
    """Run the kriging interpolation with the specified mode."""
    print(f"\n{'='*80}")
    print(f"Running with mode: {mode}")
    print(f"{'='*80}\n")

    # Create a custom namespace to simulate command-line arguments
    class Args:
        pass

    args = Args()
    args.mode = mode
    args.points = points
    args.grid_size = grid_size
    args.processes = processes
    args.idw_power = idw_power

    # Override sys.argv to prevent argparse from reading command-line arguments
    import sys
    old_argv = sys.argv
    sys.argv = [old_argv[0]]

    try:
        # Run the main function with the specified mode
        main(args)
    finally:
        # Restore sys.argv
        sys.argv = old_argv

if __name__ == "__main__":
    import sys

    # Check if command-line arguments are provided
    if len(sys.argv) > 1:
        # If arguments are provided, run with those arguments
        main()
    else:
        modes_to_test = ['mp', 'gpu', 'idw']  # Include 'gpu' only if available

        # Loop over each mode and test with 25 points and 100 points
        for mode in modes_to_test:
            for point_count in [25, 100]:
                print(f"\nRunning mode='{mode}' with {point_count} points:")
                run_with_mode(mode=mode, points=point_count, grid_size=50)

        print("\nAll mode tests completed!")
