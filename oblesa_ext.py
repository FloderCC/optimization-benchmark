"""
Implementation of various optimization methods for oblesa population initialization
    - ESS (default)
    - Latin Hypercube Sampling
    - Sobol Sequences
"""
import joblib
import numpy as np
from ess import ess
import random as rnd
from pyBlindOpt import utils
from pyBlindOpt.init import random
from scipy.stats import qmc

# adapted from: https://github.com/sparks-baird/self-driving-lab-demo/blob/main/notebooks/escience/1.0-traditional-doe-vs-bayesian.ipynb

# pip install "git+https://github.com/mariolpantunes/pyBlindOpt.git@main#egg=pyBlindOpt"

def latin_hypercube_samples(bounds, n_pop=10, seed=None):
    sampler = qmc.LatinHypercube(d=len(bounds), optimization="random-cd", seed=seed)
    samples = sampler.random(n_pop)
    l_bounds = [bound[0] for bound in bounds]
    u_bounds = [bound[1] for bound in bounds]
    samples = qmc.scale(samples, l_bounds, u_bounds)
    return samples

def sobol_samples(bounds, n_pop=10, seed=None):
    sampler = qmc.Sobol(len(bounds), seed=seed)
    samples = sampler.random(n_pop)
    l_bounds = [bound[0] for bound in bounds]
    u_bounds = [bound[1] for bound in bounds]
    samples = qmc.scale(samples, l_bounds, u_bounds)
    return samples

def essa_samples(bounds, n_pop=10, ess_pop_ratio=0.7, epochs=64, lr=0.01, k='auto', seed=None):
    # get a random initial population
    rnd_pop_size = int(n_pop * (1 - ess_pop_ratio))
    random_population = random(bounds=bounds, n_pop=rnd_pop_size, seed=seed)
    # compute the empty space population
    empty_population = ess.esa(random_population, bounds, n=n_pop - rnd_pop_size, epochs=epochs, lr=lr, k=k, seed=seed)
    return empty_population


def oblesa_ext(objective: callable, bounds: np.ndarray,
           n_pop: int = 30, n_jobs: int = -1, epochs: int = 64,
           lr: float = 0.01, k='auto', seed: int | None = None, ess_rep: str = "ess"):
    # set the random seed
    if seed is not None:
        np.random.seed(seed)
        rnd.seed(seed)  # <-- this was missing

    # get a initial random population
    random_population = random(bounds=bounds, n_pop=n_pop, seed=seed)

    # compute the fitness of the initial population
    random_scores = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective)(c) for c in random_population)
    # random_scores = utils.compute_objective(random_population, objective, n_jobs)
    # compute the opposition population
    a = bounds[:, 0]
    b = bounds[:, 1]
    opposition_population = [a + b - p for p in random_population]

    # compute the fitness of the opposition population
    # opposition_scores = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective)(c) for c in opposition_population)
    opposition_scores = utils.compute_objective(opposition_population, objective, n_jobs)

    if ess_rep == "ess":
        # computes the empty space population
        samples = np.concatenate((random_population, opposition_population), axis=0)
        empty_population = ess.esa(samples, bounds, n=n_pop, epochs=epochs, lr=lr, k=k, seed=seed)
    elif ess_rep == "lhs":
        empty_population = latin_hypercube_samples(bounds, n_pop=n_pop, seed=seed)
    elif ess_rep == "sobol":
        empty_population = sobol_samples(bounds, n_pop=n_pop, seed=seed)
    else:
        raise ValueError(f"Unknown ess_rep method: {ess_rep}. Supported methods are 'ess', 'lhs', 'sobol'.")

    # compute the fitness of the empty population
    # empty_scores = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(objective)(c) for c in empty_population)
    empty_scores = utils.compute_objective(empty_population, objective, n_jobs)

    # merge all scores and populations
    # scores = random_scores + opposition_scores + empty_scores
    # population = np.concatenate((random_population, opposition_population, empty_population), axis=0)

    population = np.concatenate((random_population, opposition_population, empty_population), axis=0)
    scores = np.concatenate((random_scores, opposition_scores, empty_scores), axis=0)

    probs = utils.score_2_probs(scores)

    return np.array(rnd.choices(population=population, weights=probs, k=n_pop))