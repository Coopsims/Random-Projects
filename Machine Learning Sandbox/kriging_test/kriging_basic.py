import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial.distance import cdist
from scipy.interpolate import Rbf
from sklearn.linear_model import LinearRegression
import random
import time
from multiprocessing import Pool, cpu_count

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
    variation = np.random.normal(0, 10, n)
    temperatures = pattern + variation

    # Clip to reasonable temperature range
    temperatures = np.clip(temperatures, 20, 90)  # Lower minimum temperature to allow for colder mountains

    return longitudes, latitudes, temperatures

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

def run_kriging_comparison(n_points=100, grid_size=50, n_jobs=None, methods=None):
    """
    Run a comparison of kriging and interpolation methods.

    Args:
        n_points: Number of random data points to generate
        grid_size: Size of the interpolation grid
        n_jobs: Number of processes to use for multiprocessing
        methods: List of methods to compare. If None, use all available methods.
                 Options: 'ordinary_kriging', 'universal_kriging', 'regression_kriging',
                 'indicator_kriging', 'rbf', 'idw'

    Returns:
        Dictionary with results for each method
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Generate random temperature data points across Colorado
    print(f"Generating random temperature data points across Colorado (n={n_points})...")
    longitudes, latitudes, temperatures = generate_random_points(n_points)

    # Create a grid for interpolation
    print(f"Creating grid for interpolation ({grid_size}x{grid_size})...")
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(CO_WEST, CO_EAST, grid_size),
        np.linspace(CO_SOUTH, CO_NORTH, grid_size)
    )

    # Prepare data for interpolation
    points = np.column_stack((longitudes, latitudes))
    grid_points = np.column_stack((grid_lon.flatten(), grid_lat.flatten()))

    # Calculate experimental variogram
    print("Calculating experimental variogram...")
    bin_centers, gamma = calculate_variogram(points, temperatures)

    # Fit variogram model
    print("Fitting variogram model...")
    variogram_params = fit_variogram_model(bin_centers, gamma)
    print(f"Variogram parameters: {variogram_params}")

    # Define available methods
    available_methods = {
        'ordinary_kriging': {
            'function': ordinary_kriging_mp,
            'params': {'variogram_params': variogram_params, 'n_jobs': n_jobs},
            'display_name': 'Ordinary Kriging (MP)'
        },
        'universal_kriging': {
            'function': universal_kriging,
            'params': {'variogram_params': variogram_params, 'trend_degree': 1, 'n_jobs': n_jobs},
            'display_name': 'Universal Kriging'
        },
        'regression_kriging': {
            'function': regression_kriging,
            'params': {'variogram_params': variogram_params, 'n_jobs': n_jobs},
            'display_name': 'Regression Kriging'
        },
        'indicator_kriging': {
            'function': indicator_kriging,
            'params': {'variogram_params': variogram_params, 'threshold': 60, 'n_jobs': n_jobs},
            'display_name': 'Indicator Kriging'
        },
        'rbf': {
            'function': radial_basis_function,
            'params': {'function': 'multiquadric', 'epsilon': 1},
            'display_name': 'Radial Basis Function'
        },
        'idw': {
            'function': inverse_distance_weighting,
            'params': {'power': 5.0},
            'display_name': 'Inverse Distance Weighting'
        }
    }

    # Determine which methods to run
    if methods is None:
        # Use all methods except indicator_kriging by default
        methods = ['ordinary_kriging', 'universal_kriging', 'regression_kriging', 'rbf', 'idw']

    # Run selected methods
    results = {}
    for method_name in methods:
        if method_name not in available_methods:
            print(f"Warning: Unknown method '{method_name}'. Skipping.")
            continue

        method = available_methods[method_name]
        print(f"Performing {method['display_name']}...")

        start_time = time.time()
        result = method['function'](points, temperatures, grid_points, **method['params'])
        execution_time = time.time() - start_time

        print(f"{method['display_name']} completed in {execution_time:.2f} seconds")

        # Reshape result to match the grid
        result = result.reshape(grid_lon.shape)

        # Store result
        results[method_name] = {
            'result': result,
            'time': execution_time,
            'display_name': method['display_name']
        }

    # Plot the results
    print("Plotting results...")

    # Calculate the aspect ratio for Colorado
    lon_range = CO_EAST - CO_WEST
    lat_range = CO_NORTH - CO_SOUTH
    aspect_ratio = np.cos(np.radians((CO_NORTH + CO_SOUTH) / 2))

    # Determine number of rows and columns for subplots
    n_methods = len(results)
    n_plots = n_methods + 1  # +1 for original data
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    # Create figure
    fig = plt.figure(figsize=(6*n_cols, 5*n_rows))

    # Plot 1: Original data points
    ax = fig.add_subplot(n_rows, n_cols, 1)
    scatter = ax.scatter(longitudes, latitudes, c=temperatures, cmap='coolwarm', 
                       s=50, edgecolor='k', norm=Normalize(vmin=30, vmax=90))
    plt.colorbar(scatter, ax=ax, label='Temperature (°F)')
    ax.set_title(f'Original Data Points (n={n_points})')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(CO_WEST, CO_EAST)
    ax.set_ylim(CO_SOUTH, CO_NORTH)
    ax.grid(True)
    ax.set_aspect(aspect_ratio)

    # Plot each interpolation method
    for i, (method_name, result_data) in enumerate(results.items()):
        # Create subplot
        ax = fig.add_subplot(n_rows, n_cols, i+2)

        # Plot interpolated surface
        contour = ax.contourf(grid_lon, grid_lat, result_data['result'], cmap='coolwarm', 
                            levels=20, norm=Normalize(vmin=30, vmax=90))
        plt.colorbar(contour, ax=ax, label='Temperature (°F)')

        # Add original data points
        ax.scatter(longitudes, latitudes, c='k', s=10, alpha=0.5)

        # Set title and labels
        ax.set_title(f"{result_data['display_name']}\nTime: {result_data['time']:.2f}s")
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xlim(CO_WEST, CO_EAST)
        ax.set_ylim(CO_SOUTH, CO_NORTH)
        ax.grid(True)
        ax.set_aspect(aspect_ratio)

    plt.tight_layout()
    plt.savefig('kriging_comparison.png', dpi=300)
    plt.show()

    return results

if __name__ == "__main__":
    # This block will run when the file is executed directly
    # It won't run when the file is imported as a module
    print("Running kriging comparison...")

    # Run comparison with default methods
    # You can specify which methods to run by passing a list to the methods parameter
    # Available methods: 'ordinary_kriging', 'universal_kriging', 'regression_kriging', 'indicator_kriging', 'rbf', 'idw'
    results = run_kriging_comparison(n_points=100, grid_size=200)

    print("\nComparison completed!")

    # Print execution times for all methods
    print("\nExecution times:")
    for method_name, result_data in results.items():
        print(f"{result_data['display_name']}: {result_data['time']:.2f} seconds")
