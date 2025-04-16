# Kriging and IDW Interpolation with GPU and Multiprocessing Support

This script performs spatial interpolation on randomly generated temperature data across Colorado. It supports four processing modes:

1. **CPU** - Default mode, uses single-core processing for kriging
2. **Multiprocessing** - Distributes the kriging workload across multiple CPU cores
3. **GPU** - Uses GPU acceleration via PyTorch for kriging (supports both CUDA for NVIDIA GPUs and MPS for Apple Silicon)
4. **IDW** - Uses Inverse Distance Weighting, a faster alternative to kriging

## Features

- Semi-normal distribution of data points across longitude
- Realistic temperature pattern (warm-cool-warm from west to east)
- Proper aspect ratio for Colorado map visualization
- Multiple interpolation methods with various acceleration options
- Reproducible results with fixed random seed (42)

## Requirements

- Python 3.6+
- NumPy
- SciPy
- Matplotlib
- PyTorch (optional, for GPU acceleration)

## Usage

Run the script with the following command-line arguments:

```bash
python kriging.py [--mode {cpu,mp,gpu,idw}] [--processes PROCESSES] [--points POINTS] [--grid-size GRID_SIZE] [--idw-power IDW_POWER]
```

### Arguments

- `--mode`: Processing mode (default: 'cpu')
  - `cpu`: CPU processing for kriging using a single core
  - `mp`: Multiprocessing (parallel CPU processing) for kriging
  - `gpu`: GPU acceleration using PyTorch for kriging
  - `idw`: Inverse Distance Weighting (faster alternative to kriging)

- `--processes`: Number of processes to use for multiprocessing (default: number of CPU cores)

- `--points`: Number of random data points to generate (default: 100)

- `--grid-size`: Size of the interpolation grid (default: 50x50)

- `--idw-power`: Power parameter for IDW interpolation (default: 2.0). Higher values give more weight to closer points.

### Examples

Run with default settings (CPU mode):
```bash
python kriging.py
```

Run with multiprocessing using 4 processes:
```bash
python kriging.py --mode mp --processes 4
```

Run with GPU acceleration:
```bash
python kriging.py --mode gpu
```

Run with the faster IDW method:
```bash
python kriging.py --mode idw
```

Run IDW with a different power parameter:
```bash
python kriging.py --mode idw --idw-power 3.0
```

Generate more data points and use a larger grid:
```bash
python kriging.py --mode gpu --points 200 --grid-size 100
```

## Performance Considerations

- **IDW mode** is the fastest option and provides good results for most applications.
- **CPU mode** uses a single core for kriging, which is useful for benchmarking or when you want to minimize resource usage.
- **Multiprocessing mode** uses all available CPU cores for kriging, providing better performance on multi-core systems.
- **GPU mode** provides the best performance for large grids through optimized batch processing:
  - Uses batched linear algebra operations to solve multiple kriging systems simultaneously
  - Processes grid points in configurable batches to optimize memory usage
  - Employs fully vectorized operations to maximize GPU utilization
  - Minimizes CPU-GPU data transfers for better performance
  - Particularly effective for large grids (100x100 or larger) and complex interpolation tasks

### Apple Silicon (MPS) Optimizations

The script includes special optimizations for Apple Silicon Macs using Metal Performance Shaders (MPS):
- Uses LU factorization when available for more stable matrix solving
- Applies numerical stability enhancements with regularization
- Processes data in smaller batches to avoid memory issues
- Provides graceful fallback to IDW when necessary
- Includes detailed error reporting for troubleshooting

The script will automatically:
- Fall back to multiprocessing if GPU acceleration is requested but PyTorch is not available
- Use Apple's Metal Performance Shaders (MPS) on Apple Silicon Macs if CUDA is not available
- Adjust batch sizes based on available GPU memory to prevent out-of-memory errors
- Select the most appropriate solving method based on the hardware

## Data Distribution and Temperature Pattern

- Points are distributed with a semi-normal distribution across longitude, with more points concentrated in the center of Colorado
- Temperatures follow a warm-cool-warm pattern from west to east, with the coolest temperatures around longitude -107.0 to -106.5
- The mountain regions (west of longitude -105.5) are significantly colder, with temperatures decreasing as you move further west into the mountains
- Random variation is added to make the temperature pattern more realistic

## Output

The script generates a visualization of the original data points and the interpolation results, with proper aspect ratio to maintain the correct shape of Colorado. The output is saved as a PNG file:

- For kriging: `colorado_temperature_kriging_<mode>.png`
- For IDW: `colorado_temperature_idw_power<power>.png`
