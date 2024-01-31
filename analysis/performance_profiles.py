""" 
Reads the results from data/results
and produces the performance profiles as documenated in the paper.
"""

from utils import make_performance_profile

for optimizer in ["both", "CPLEX", "Gurobi"]:
    make_performance_profile(
        "CDP",
        "data/results/fcdp-1-thread.csv",
        optimizer=optimizer,
        cutoff=5,
        save=True,
    )
    make_performance_profile(
        "CDP with 16 threads",
        "data/results/fcdp-16-thread.csv",
        optimizer=optimizer,
        cutoff=5,
        save=True,
    )
    make_performance_profile(
        "RCDP", "data/results/rcdp-1-thread.csv", optimizer=optimizer, save=True
    )

    make_performance_profile(
        "GDP",
        "data/results/fgdp-1-thread.csv",
        optimizer=optimizer,
        cutoff=5,
        save=True,
    )
    make_performance_profile(
        "GDP with 16 threads",
        "data/results/fgdp-16-thread.csv",
        optimizer=optimizer,
        cutoff=5,
        save=True,
    )
    make_performance_profile(
        "RGDP", "data/results/rgdp-1-thread.csv", optimizer=optimizer, save=True
    )

    make_performance_profile(
        "BDP-D",
        "data/results/fbdp-d-1-thread.csv",
        optimizer=optimizer,
        cutoff=70,
        save=True,
    )
    make_performance_profile(
        "DP-C",
        "data/results/fdp-c-1-thread.csv",
        optimizer=optimizer,
        save=True,
    )
