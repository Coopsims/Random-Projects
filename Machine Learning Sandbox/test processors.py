import multiprocessing
import psutil
import time


def cpu_bound_task(duration):
    # CPU-bound loop for a given duration
    end_time = time.time() + duration
    while time.time() < end_time:
        pass  # Busy-wait


if __name__ == "__main__":
    duration = 10  # Duration of the stress test in seconds
    n_jobs = psutil.cpu_count(logical=True)
    print("Starting stress test on", n_jobs, "logical cores.")

    # Create a pool to run the CPU-bound tasks concurrently
    pool = multiprocessing.Pool(processes=n_jobs)
    tasks = [pool.apply_async(cpu_bound_task, args=(duration,)) for _ in range(n_jobs)]

    # Monitor per-core CPU usage during the stress test
    for _ in range(duration):
        cpu_percentages = psutil.cpu_percent(interval=1, percpu=True)
        print("CPU usage per core:", cpu_percentages)

    pool.close()
    pool.join()