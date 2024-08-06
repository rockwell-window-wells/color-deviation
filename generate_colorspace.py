import numpy as np
import matplotlib.pyplot as plt
from colorspacious import cspace_convert, deltaE
import random

# Define the dimensions of the grid
m, n = 5, 5  # Example grid size; you can change this

# Define the reference color in Lab space
lab_reference = np.array([79.43, 5.33, 1.47])

# Convert the reference color to RGB
rgb_reference = cspace_convert(lab_reference, "CIELab", "sRGB1")

# Parameters for the Genetic Algorithm
population_size = 50
num_generations = 100
mutation_rate = 0.1

# Function to generate random valid Lab colors
def generate_random_valid_lab_color():
    while True:
        L = np.random.uniform(0, 100)
        a = np.random.uniform(-128, 127)
        b = np.random.uniform(-128, 127)
        lab_color = np.array([L, a, b])
        try:
            rgb_color = cspace_convert(lab_color, "CIELab", "sRGB1")
            if (rgb_color >= 0).all() and (rgb_color <= 1).all():
                return lab_color
        except ValueError:
            continue

# Function to generate initial population
def generate_initial_population(population_size, num_colors):
    population = []
    for _ in range(population_size):
        individual = [generate_random_valid_lab_color() for _ in range(num_colors)]
        population.append(individual)
    return population

# Function to calculate the average deltaE between colors in a set
def average_deltaE(colors):
    n = len(colors)
    deltaE_sum = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            deltaE_sum += deltaE(colors[i], colors[j])
            count += 1
    return deltaE_sum / count

# Evaluate fitness of each individual in the population
def evaluate_population(population):
    fitness_scores = []
    for individual in population:
        fitness_scores.append(average_deltaE(individual))
    return fitness_scores

# Selection: Tournament selection
def select_parents(population, fitness_scores):
    parents = []
    for _ in range(len(population) // 2):
        i, j = np.random.choice(len(population), 2, replace=False)
        if fitness_scores[i] > fitness_scores[j]:
            parents.append(population[i])
        else:
            parents.append(population[j])
    # Ensure the number of parents is even
    if len(parents) % 2 != 0:
        parents.append(parents[0])
    return parents

# Crossover: Single-point crossover
def crossover(parents):
    next_generation = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i+1]
        cross_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:cross_point] + parent2[cross_point:]
        child2 = parent2[:cross_point] + parent1[cross_point:]
        next_generation.extend([child1, child2])
    return next_generation

# Mutation: Randomly change one Lab color in the individual
def mutate(individual, mutation_rate):
    if np.random.rand() < mutation_rate:
        idx = np.random.randint(len(individual))
        individual[idx] = generate_random_valid_lab_color()
    return individual

# Genetic Algorithm
num_colors = m * n - 4
population = generate_initial_population(population_size, num_colors)
best_individual = None
best_fitness = -np.inf

for generation in range(num_generations):
    fitness_scores = evaluate_population(population)
    best_generation_fitness = max(fitness_scores)
    if best_generation_fitness > best_fitness:
        best_fitness = best_generation_fitness
        best_individual = population[np.argmax(fitness_scores)]
    
    parents = select_parents(population, fitness_scores)
    next_generation = crossover(parents)
    population = [mutate(individual, mutation_rate) for individual in next_generation]

# Convert the best Lab colors to RGB
rgb_colors = np.array([cspace_convert(color, "CIELab", "sRGB1") for color in best_individual])

# Create the grid
grid = np.zeros((m, n, 3))

# Place the reference color in the center four squares
center_positions = [(m // 2 - 1, n // 2 - 1), (m // 2 - 1, n // 2), (m // 2, n // 2 - 1), (m // 2, n // 2)]
for pos in center_positions:
    grid[pos] = rgb_reference

# Fill the rest of the grid with selected colors
color_index = 0
for i in range(m):
    for j in range(n):
        if grid[i, j].sum() == 0:  # Skip the center four squares
            grid[i, j] = rgb_colors[color_index]
            color_index += 1

# Plot the grid
plt.figure(figsize=(n, m))
plt.imshow(grid, aspect='equal')
plt.axis('off')
plt.show()
