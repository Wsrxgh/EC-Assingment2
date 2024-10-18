import sys
import numpy as np
import os
import random
import pandas as pd
from evoman.environment import Environment
from demo_controller import player_controller
import time

os.environ["SDL_VIDEODRIVER"] = "dummy"

n_hidden_neurons = 10
enemies_group1 = [1, 2, 5]  
enemies_group2 = [4, 6, 7]  

pop_size = 150
gens = 60
n_runs = 10

def calculate_n_weights(n_inputs, n_hidden_neurons):
    return (n_inputs + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

# Get fitness
def fitness_function(fit, p_energy, e_energy, time):
    return fit
 
# Calculate gain
def calculate_gain(p_energy, e_energy):
    return p_energy - e_energy

# Simulation function
def simulation(env, x):
    fit, p_energy, e_energy, time = env.play(pcont=x)
    gain = calculate_gain(p_energy, e_energy)
    return fit, gain

# Initialize population
def initialize_population(pop_size, n_weights):
    return np.random.uniform(-1, 1, (pop_size, n_weights))

# Evaluate population fitness and gain
def evaluate_population(population, env):
    fitness_scores = []
    gain_scores = []
    for x in population:
        fitness, gain = simulation(env, x)
        fitness_scores.append(fitness)
        gain_scores.append(gain)
    return np.array(fitness_scores), np.array(gain_scores)

# Differential Evolution
def mutation_and_crossover(population, target_idx, gen, gens, n_weights):
    F = 0.5
    CR = 0.9

    indices = list(range(pop_size))
    indices.remove(target_idx)
    a, b, c = random.sample(indices, 3)

    mutant = population[a] + F * (population[b] - population[c])
    mutant = np.clip(mutant, -1, 1)

    trial = np.copy(population[target_idx])
    for i in range(n_weights):
        if random.random() < CR or i == random.randint(0, n_weights - 1):
            trial[i] = mutant[i]

    return trial

# Main Loop
def run_differential_evolution(env, enemies, run_idx, indices_run, indices_gen, best_gain, best_fit, mean_fitness, std_fitness, best_solutions):
    n_inputs = env.get_num_sensors()  
    n_weights = calculate_n_weights(n_inputs, n_hidden_neurons)

    population = initialize_population(pop_size, n_weights)
    fitness_scores, gain_scores = evaluate_population(population, env)

    for gen in range(gens):
        new_population = np.copy(population)

        # Elite preservation
        best_idx = np.argmax(fitness_scores)
        new_population[0] = population[best_idx]
        fitness_scores[0] = fitness_scores[best_idx]
        gain_scores[0] = gain_scores[best_idx]

        for i in range(1, pop_size):  
            trial = mutation_and_crossover(population, i, gen, gens, n_weights)
            trial_fitness, trial_gain = simulation(env, trial)

            if trial_fitness > fitness_scores[i]:
                new_population[i] = trial
                fitness_scores[i] = trial_fitness
                gain_scores[i] = trial_gain

        population = new_population
        best_idx = np.argmax(fitness_scores)
        avg_fitness = np.mean(fitness_scores)
        max_fitness = np.max(fitness_scores)
        std_fitness_val = np.std(fitness_scores)
        best_weights = population[best_idx]

        # Record results for each generation
        indices_run.append(run_idx)
        indices_gen.append(gen)
        best_gain.append(gain_scores[best_idx])
        best_fit.append(max_fitness)
        mean_fitness.append(avg_fitness)
        std_fitness.append(std_fitness_val)
        best_solutions.append(best_weights.tolist())
        print(f'RUN {run_idx}, GENERATION {gen} - Best Fitness: {max_fitness}, Gain: {gain_scores[best_idx]}')

def run_experiments():
    if not os.path.exists('results'):
        os.makedirs('results')

    experiment_names = ["EA2_enemylist1", "EA2_enemylist2"]
    for enemies, experiment_name in zip([enemies_group1, enemies_group2], experiment_names):
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        env = Environment(experiment_name=experiment_name,
                          enemies=enemies,
                          playermode="ai",
                          player_controller=player_controller(n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=False,
                          multiplemode='yes'
                          )

        indices_run = []
        indices_gen = []
        best_gain = []
        best_fit = []
        mean_fitness = []
        std_fitness = []
        best_solutions = []

        print(f"Running Differential Evolution on enemies: {enemies}")
        for run_idx in range(n_runs):
            run_differential_evolution(env, enemies, run_idx, indices_run, indices_gen, best_gain, best_fit, mean_fitness, std_fitness, best_solutions)

        d = {"Run": indices_run, "Gen": indices_gen, "gain": best_gain, "Best fit": best_fit, "Mean": mean_fitness, "STD": std_fitness, "BEST SOL": best_solutions}
        df = pd.DataFrame(data=d)
        df.to_csv(f'{experiment_name}/{experiment_name}.csv', index=False)

if __name__ == "__main__":
    run_experiments()
