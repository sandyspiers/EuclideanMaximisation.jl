import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from multiprocessing import Pool

from data.instance_handler import (
    capacitated_diversity_problem_from_file,
    generalized_diversity_problem_from_file,
    random_capacitated_diversity_problem,
    random_generalized_diversity_problem,
)

from emsca.model import EmsModel

LOG_FILE = "results/results.log"


def record_results(mdl: EmsModel):
    with open(LOG_FILE, "a") as f:
        f.write(
            f"{mdl.name},{mdl._original_timelimit},{mdl.solver},{mdl.add_lp_tangents},"
        )
        f.write(
            f"{mdl.solve_details.time},{mdl.objective_value},{mdl.solve_details.best_bound},"
        )
        f.write(f"{mdl.iterations},{mdl.lp_cut_counter},{mdl.ip_cut_counter}\n")


def solve_with_all_solvers(instance_generator, *args):
    mdl: EmsModel
    for s, rc in EmsModel.ALL_SOLVER_OPTIONS:
        with instance_generator(*args) as mdl:
            mdl.verbose = 0
            mdl.solver = s
            mdl.add_lp_tangents = rc
            mdl.name = f"{mdl.name}_{s}_{rc}"
            mdl.parameters.timelimit = 600
            mdl.solve()
            record_results(mdl)


def solve_with_best_solvers(instance_generator, *args):
    mdl: EmsModel
    solvers = ["repoa", "fcard"]
    rootcut = [True, "rootonly", False]
    for s in solvers:
        for rc in rootcut:
            with instance_generator(*args) as mdl:
                mdl.verbose = 0
                mdl.solver = s
                mdl.add_lp_tangents = rc
                mdl.name = f"{mdl.name}_{s}_{rc}"
                mdl.parameters.timelimit = 600
                mdl.solve()
                record_results(mdl)


def run_CDP_tests(directory):
    pool = Pool()
    for file in os.listdir(directory):
        if file.__contains__("GKD"):
            pool.apply_async(
                solve_with_all_solvers,
                (capacitated_diversity_problem_from_file, f"{directory}/{file}"),
            )
    pool.close()
    pool.join()


def run_GDP_tests(directory):
    pool = Pool()
    for file in os.listdir(directory):
        if file.__contains__("GKD"):
            pool.apply_async(
                solve_with_all_solvers,
                (generalized_diversity_problem_from_file, f"{directory}/{file}"),
            )
    pool.close()
    pool.join()


def run_RCDP_tests():
    seed = 0
    pool = Pool()
    for num in [1000, 1500, 2000, 2500, 3000]:
        for coords in [2, 10, 20]:
            for capacity_ratio in [0.2, 0.3]:
                for k in range(5):
                    seed += 1
                    pool.apply_async(
                        solve_with_best_solvers,
                        (
                            random_capacitated_diversity_problem,
                            num,
                            coords,
                            capacity_ratio,
                            seed,
                        ),
                    )
    pool.close()
    pool.join()


def run_RGDP_tests():
    seed = 0
    pool = Pool()
    for num in [1000, 1500, 2000, 2500, 3000]:
        for coords in [2, 10, 20]:
            for capacity_ratio in [0.2, 0.3]:
                for cost_ratio in [0.5, 0.6]:
                    for k in range(5):
                        seed += 1
                        pool.apply_async(
                            solve_with_best_solvers,
                            (
                                random_generalized_diversity_problem,
                                num,
                                coords,
                                capacity_ratio,
                                cost_ratio,
                                seed,
                            ),
                        )
    pool.close()
    pool.join()


if __name__ == "__main__":
    run_CDP_tests("results/data/CDP")
    run_GDP_tests("results/data/GDP")
    run_RCDP_tests()
    run_RGDP_tests()
