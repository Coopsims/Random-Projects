import random
from tqdm import tqdm

num_seeds = 1000000
num_draws = 365  # The number of random values per seed

total_collisions = 0
total_percentage = 0.0

for seed_val in tqdm(range(num_seeds)):
    random.seed(seed_val)
    random_numbers = [int(random.random() * 365) for _ in range(num_draws)]

    collisions = len(random_numbers) - len(set(random_numbers))
    collision_percentage = (collisions / num_draws) * 100

    total_collisions += collisions
    total_percentage += collision_percentage

average_collisions = total_collisions / num_seeds
average_collision_percentage = total_percentage / num_seeds

print(f"Average collisions: {average_collisions:.2f}")
print(f"Average collision percentage: {average_collision_percentage:.6f}%")
