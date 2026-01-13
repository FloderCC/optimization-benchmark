import os
import time

import numpy as np
import pandas as pd
import cocoex
from pyBlindOpt.de import differential_evolution
from pyBlindOpt.ga import genetic_algorithm
from pyBlindOpt.gwo import grey_wolf_optimization
from pyBlindOpt.egwo import grey_wolf_optimization as e_grey_wolf_optimization
from pyBlindOpt.init import random, opposition_based
from pyBlindOpt.pso import particle_swarm_optimization

from oblesa_ext import oblesa_ext, latin_hypercube_samples, sobol_samples

# =========================
# CONFIG
# =========================
SUITE_NAME = "bbob"

# NOTE: Suite(name, instance, options)
SUITE_INSTANCE = "instances: 1-3"      # change this if you want more/fewer instances
SUITE_OPTIONS  = "dimensions: 2,5,10,20,40"      # fix the dimension to compare “like with like” 20, 40

N_POP = 64          # population size 64, 128, 256
# SEEDS = [3, 5, 7, 11, 13, 17, 42]
SEEDS = [5, 7, 42]
N_REPS = len(SEEDS)

OPT_N_ITER = 3000 # 3000
TOL = 10e-8

N_JOBS = 1

INIT_METHODS = [
    "random",
    "lhs",
    "sobol",
    "obl",
    "oblesa_ess",
    "oblesa_lhs",
    "oblesa_sobol",
    "oblesa_sobol"
]

OPT_METHODS = [
    "ga",
    "de",
    "pso",
    "gwo",
    "egwo"
]

# =========================
# Helpers
# =========================
def get_bounds(problem):
    lb = np.asarray(problem.lower_bounds, dtype=float)
    ub = np.asarray(problem.upper_bounds, dtype=float)
    return np.stack([lb, ub], axis=1)  # (d,2)


def get_fopt_via_best_parameter(problem):
    tmp_file_name = "._bbob_problem_best_parameter.txt"
    try:
        problem._best_parameter("print")
        xopt = np.loadtxt(tmp_file_name).reshape(-1)
        fopt = float(problem(xopt))
        return fopt
    finally:
        try: os.remove(tmp_file_name)
        except OSError: pass
        pass

def generate_population(method, problem, bounds, n_pop, seed):
    if method == "random":
        return random(bounds, n_pop, seed)
    if method == "lhs":
        return latin_hypercube_samples(bounds, n_pop=n_pop, seed=seed)
    if method == "sobol":
        return sobol_samples(bounds, n_pop=n_pop, seed=seed)
    if method == "obl":
        return opposition_based(problem, bounds, n_pop=n_pop, seed=seed, n_jobs=N_JOBS)

    if method == "oblesa_ess":
        return oblesa_ext(problem, bounds, n_pop=n_pop, seed=seed, ess_rep="ess", n_jobs=N_JOBS)
    if method == "oblesa_lhs":
        return oblesa_ext(problem, bounds, n_pop=n_pop, seed=seed, ess_rep="lhs", n_jobs=N_JOBS)
    if method == "oblesa_sobol":
        return oblesa_ext(problem, bounds, n_pop=n_pop, seed=seed, ess_rep="sobol", n_jobs=N_JOBS)

    raise ValueError(f"Unknown method: {method}")

def eval_population(problem, X):
    # Evaluate point-by-point (COCO counts evaluations internally; that doesn’t matter here)
    vals = np.array([problem(x) for x in X], dtype=float)
    return vals

def make_train_stop_callback(fopt, tol=1e-3):
    def callback(epoch, scores, pop) -> bool:
        best_score = min(scores)
        return (best_score - fopt) < tol
    return callback

def train_opt_method(method, problem, X_init, n_iter=OPT_N_ITER, seed=42, n_jobs=N_JOBS, debug=True, callback=None):
    pop_list = X_init
    bounds = get_bounds(problem)

    if method == "ga":
        best_solution = genetic_algorithm(problem, bounds=bounds, population=pop_list, n_pop=len(pop_list), seed = seed, n_iter=n_iter, n_jobs=n_jobs, debug=debug, callback=callback)
    elif method == "de":
        best_solution = differential_evolution(problem, bounds=bounds, population=pop_list, n_pop=len(pop_list), seed = seed, n_iter=n_iter, n_jobs=n_jobs, debug=debug, callback=callback)
    elif method == "pso":
        best_solution = particle_swarm_optimization(problem, bounds=bounds, population=pop_list, n_pop=len(pop_list), seed = seed, n_iter=n_iter, n_jobs=n_jobs, debug=debug, callback=callback)
    elif method == "gwo":
        best_solution = grey_wolf_optimization(problem, bounds=bounds, population=pop_list, n_pop=len(pop_list), seed = seed, n_iter=n_iter, n_jobs=n_jobs, debug=debug, callback=callback)
    elif method == "egwo":
        best_solution = e_grey_wolf_optimization(problem, bounds=bounds, population=pop_list, n_pop=len(pop_list), seed = seed, n_iter=n_iter, n_jobs=n_jobs, debug=debug, callback=callback)
    else:
        raise ValueError(f"Unknown optimization method: {method}")
    return best_solution

# =========================
# Main experiment
# =========================
def run():
    suite = cocoex.Suite(SUITE_NAME, SUITE_INSTANCE, SUITE_OPTIONS)
    dims = [p.dimension for p in suite]
    print("dims present:", sorted(set(dims)))
    print("counts:", {d: dims.count(d) for d in sorted(set(dims))})

    rows = []
    conf_qty = len(suite) * len(SEEDS) * len(INIT_METHODS) * len(OPT_METHODS)
    conf_n = 0
    print(f"Running experiment on suite '{SUITE_NAME}' with {len(suite)} problems, {N_REPS} reps, {len(INIT_METHODS)} methods -> total setups: {conf_qty}")
    for problem in suite:
        bounds = get_bounds(problem)
        fopt = get_fopt_via_best_parameter(problem)
        cb = make_train_stop_callback(fopt, tol=TOL)

        pid = getattr(problem, "id", None) or str(problem)

        for r, seed in enumerate(SEEDS):
            # For a fair comparison, use the SAME base seed for each method.
            for init_method in INIT_METHODS:
                tic = time.time()
                X = generate_population(init_method, problem, bounds, N_POP, seed)
                toc = time.time()
                init_time = toc - tic

                for opt_method in OPT_METHODS:
                    print(f"Running config {conf_n+1}/{conf_qty}: problem={pid}, seed={seed}, init_method={init_method}, opt_method={opt_method}")

                    # print("best ind before opt:", np.min(eval_population(problem, np.asarray(X, dtype=float))))

                    tic = time.time()
                    (_, best_eval, (obj_best_iter, _, _)) = train_opt_method(opt_method, problem, X, seed=seed, callback=cb)
                    toc = time.time()
                    opt_time = toc - tic

                    # print("best ind after opt:", best_eval)

                    best_delta = best_eval - fopt

                    # print(f"Result: best_eval={best_eval}, fopt={fopt}, best_delta={best_delta}")

                    row = {
                        "problem": pid,
                        "dim": int(problem.dimension),
                        "seed": seed,
                        "init_method": init_method,
                        "opt_method": opt_method,
                        "best_target": fopt,
                        "best_eval": best_eval,
                        "best_n_iter": len(obj_best_iter),
                        "best_delta": best_delta,
                        "init_time": init_time,
                        "opt_time": opt_time,
                    }
                    rows.append(row)

                    conf_n += 1

    df = pd.DataFrame(rows)
    return df


def summarize(df: pd.DataFrame, compare_by: str = "best_eval"): # hacer otro que sea por cant de iteraciones hasta target
    if compare_by not in ("best_eval", "best_delta", "best_iters"):
        raise ValueError("compare_by must be 'best_eval' or 'best_delta' or 'best_iters'")

    key = ["problem", "seed"]

    if compare_by == "best_eval":
        d = df.copy()

        # Rank per (problem, seed)
        d["rank"] = d.groupby(key)["best_eval"].rank(ascending=True, method="average")

        # Win-rate per (problem, seed) using best_eval: number of times each method was best across all problems and seeds
        winners = d.loc[d.groupby(key)["best_eval"].idxmin()][key + ["init_method"]]
        win_rate = winners["init_method"].value_counts(normalize=True).mul(100).round(2)

        # Normalized regret by observed range (robust to function scale)
        best = d.groupby(key)["best_eval"].transform("min")
        worst = d.groupby(key)["best_eval"].transform("max")
        span = (worst - best)
        d["norm_regret"] = np.where(span > 0, (d["best_eval"] - best) / span, 0.0)

        # Final summary per method
        summary = d.groupby("init_method").agg(
            mean_rank=("rank", "mean"),
            mean_norm_regret=("norm_regret", "mean"),
            mean_best_eval=("best_eval", "mean"),
        ).sort_values(["mean_rank", "mean_norm_regret"], ascending=True)

        print("\n=== SUMMARY (compare_by='best_eval' | relative, no target needed) ===")
        print(summary.round(4).to_string())

        print("\n=== WIN RATE (%) (best_eval) ===")
        print(win_rate.to_string())

    elif compare_by == "best_delta":
        # -----------------------------
        # 2) compare_by == "best_delta"
        #    (target-based) => success rates + win-rate by delta
        # -----------------------------
        d = df.copy()

        # Success-rate columns if they exist
        succ_cols = [c for c in d.columns if c.startswith("succ_le_")]

        # Win-rate per (problem, seed) using best_delta
        winners = d.loc[d.groupby(key)["best_delta"].idxmin()][key + ["init_method"]]
        win_rate = winners["init_method"].value_counts(normalize=True).mul(100).round(2)

        cols = ["best_delta", "best_eval", "mean_f", "median_f"] + succ_cols
        summary = d.groupby("init_method")[cols].agg(["mean", "std"])

        print("\n=== SUMMARY (compare_by='best_delta' | target-based) ===")
        print(summary.round(4).to_string())

        print("\n=== WIN RATE (%) (best_delta) ===")
        print(win_rate.to_string())

    else: # 'best_iters'
        d = df.copy()

        # passing to another dataframe instances where best_n_iter == OPT_N_ITER and best_eval is not close enough to target
        d_f = d[~((d["best_n_iter"] >= OPT_N_ITER) & (d["best_delta"] > TOL))]

        # Rank per (problem, seed)
        d["rank"] = d.groupby(key)["best_n_iter"].rank(ascending=True, method="average")

        # Win-rate per (problem, seed) using best_n_iter: number of times each method was best across all problems and seeds
        winners = d.loc[d.groupby(key)["best_n_iter"].idxmin()][key + ["init_method"]]
        win_rate = winners["init_method"].value_counts(normalize=True).mul(100).round(2)

        # Final summary per method
        summary = d.groupby("init_method").agg(
            mean_rank=("rank", "mean"),
            mean_best_n_iter=("best_n_iter", "mean"),
        ).sort_values(["mean_rank", "mean_best_n_iter"], ascending=True)

        print("\n=== SUMMARY (compare_by='best_n_iter') ===")
        print(summary.round(4).to_string())

        print("\n=== WIN RATE (%) (best_n_iter) ===")
        print(win_rate.to_string())

        # Additional info about fallouts
        print("\n=== ADDITIONAL INFO: Fallouts (did not reach target within max iters) ===")
        total_cases = len(d)
        fallout_counts = total_cases - len(d_f)
        print(f"Total cases: {total_cases}, Fallouts: {fallout_counts}")
        fallout_rate = (fallout_counts / total_cases) * 100
        print(f"Fallout Rate: {fallout_rate:.2f}%")
        # Breakdown by method
        fallout_by_method = d.groupby("init_method").apply(
            lambda x: total_cases - len(x[~((x["best_n_iter"] >= OPT_N_ITER) & (x["best_delta"] > TOL))])
        )
        # adding percentage
        fallout_by_method = fallout_by_method.to_frame(name="fallouts")
        fallout_by_method["fallout_rate (%)"] = (fallout_by_method["fallouts"] / total_cases * 100).round(2)
        print("\nFallouts by method:")
        print(fallout_by_method.to_string())

if __name__ == "__main__":
    tic = time.time()
    df = run()
    toc = time.time()
    elapsed = toc - tic
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"Total experiment time: {hours}h {minutes}m {seconds:.2f}s ({elapsed:.2f} seconds)")
    df.to_csv("bbob_initpop_comparison.csv", index=False)
    print("Saved: bbob_initpop_comparison.csv")

    df = pd.read_csv("bbob_initpop_comparison.csv")
    summarize(df, "best_iters")
