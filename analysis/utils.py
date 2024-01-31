import os
import pathlib
from glob import glob
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_yaml(filename) -> Dict:
    return yaml.safe_load(open(filename))


def get_file_name(filepath):
    return os.path.basename(filepath).split(".")[0]


def safe_index(lst, name, safety=0):
    try:
        return list(lst).index(name)
    except:
        return safety


def make_performance_profile(name, df_path, optimizer="both", cutoff=None, save=False):
    style_guide = load_yaml("analysis/style_guide.yml")
    df = pd.read_csv(df_path)

    # make axis
    if cutoff:
        fig, axes = plt.subplots(
            1, 2, sharey=True, width_ratios=[3, 2], figsize=(10, 6)
        )
    else:
        fig, axes = plt.subplots(1, figsize=(10, 6))
        axes = [axes]

    # plot performances
    if optimizer != "both":
        df = df[df["optimizer"] == optimizer]
        pp = get_performance_profiles(df, solver_col_name="solver")
    else:
        pp = get_performance_profiles(df, solver_col_name=["optimizer", "solver"])
    for ax in axes:
        plot_performance_profile(ax, pp, style_guide)

    # correct limits
    if cutoff:
        axes[0].set_xlim(right=cutoff)
        axes[1].set_xlim(left=cutoff)

    # add style guided legend
    style_guided_legend(axes[-1], style_guide)

    # generic titles
    fig.supxlabel("Time (sec)")
    axes[0].set_ylabel("Num solved")
    # specific title
    fig.suptitle(f"Solver performance on {name}")

    fig.tight_layout()
    if save:
        pathlib.Path("analysis/figs/").mkdir(parents=True, exist_ok=True)
        plt.savefig(
            f"analysis/figs/pp_{get_file_name(df_path)}_{optimizer}_cutoff-{cutoff}.pdf"
        )
    else:
        plt.show()
    plt.close()


def get_performance_profiles(
    df: pd.DataFrame,
    solver_col_name="solver",
    run_time_col_name="run_time",
    time_limit_col_name="timelimit",
) -> Dict:
    """Returns a performance profile of all unique solver setups within df"""
    # Get all unique sovers
    if type(solver_col_name) == list:
        df["combined_solver_col_name"] = list(
            zip(*[df[col_name] for col_name in solver_col_name])
        )
        solver_col_name = "combined_solver_col_name"
    solvers = set(df[solver_col_name])

    performance_profile = {
        "max_time": min(
            df[run_time_col_name].max() * 1.05, df[time_limit_col_name].max()
        ),
        "num_test": max(
            df[df[solver_col_name] == solver].shape[0] for solver in solvers
        ),
    }
    for solver in solvers:
        # get solver instances
        _df = df[
            (df[solver_col_name] == solver)
            & (df[run_time_col_name] <= df[time_limit_col_name])
        ]
        # start
        times = [0]
        # solve times
        times += list(_df[run_time_col_name].sort_values())
        # end time
        times += [_df[time_limit_col_name].max()]
        # step at solves, except the last
        steps = np.arange(len(times))
        steps[-1] -= 1
        performance_profile[solver] = (times, steps)
    return performance_profile


def plot_performance_profile(axes, performance_profile, style_guide={}):
    axes.set_xlim(0, performance_profile["max_time"])
    axes.set_ylim(bottom=0)

    for solver in performance_profile:
        if solver == "max_time":
            continue
        if solver == "num_test":
            num_test = performance_profile["num_test"]
            max_time = performance_profile["max_time"]
            axes.hlines(num_test, 0, max_time, colors=["grey"], linestyles="dashed")
            axes.set_ylim(top=num_test * 1.05)
            continue
        times, steps = performance_profile[solver]
        solver_kwargs = style_guide.get(str(solver), {})
        solver_kwargs.pop("short_name", None)
        axes.step(times, steps, where="post", **solver_kwargs)


def style_guided_legend(axes, style_guide):
    # sort both labels and handles by labels
    handles, labels = axes.get_legend_handles_labels()
    names = [style["label"] for style in style_guide.values()]
    order = [safe_index(names, labels[l], safety=l) for l in range(len(labels))]
    order = np.argsort(order)
    axes.legend([handles[idx] for idx in order], [labels[idx] for idx in order])


def add_n_col(df):
    if "num" in df.columns:
        df["n"] = df["num"]
    if "n" not in df.columns:
        if "filename" not in df.columns:
            raise Exception(f"Number associated column not found out of {df.columns}")
        df["n"] = df["filename"].str.extract(r"_n(\d+)_").astype(int)
    return df


def read_df(filename, optimizer="Gurobi"):
    df = pd.read_csv(filename)
    if optimizer:
        df = df[df["optimizer"] == optimizer]
    df["gap"] = (df["best_bound"] - df["obj_value"]) / df["obj_value"] * 100.0
    df["solved"] = df["gap"] <= 1e-6
    df = add_n_col(df)
    return df


def fix_col_names(table, style_guide):
    if "glov" not in table:
        table["glov"] = pd.NA
    if "quad" not in table:
        table["quad"] = pd.NA
    cols = table.columns
    order = [
        safe_index(style_guide.keys(), cols[l], safety=l) for l in range(len(cols))
    ]
    order = np.argsort(order)
    table = table[[table.columns[o] for o in order]]
    table.columns = [style_guide[col]["short_name"] for col in table.columns]
    return table


def create_pivot(filename, style_guide):
    df = read_df(filename)
    pt = df.pivot_table("run_time", ["name", "n"], "solver", "mean")
    pt = fix_col_names(pt, style_guide)
    return pt


def glob_stack(filename, style_guide):
    return pd.concat([create_pivot(file, style_guide) for file in glob(filename)])


def to_latex(df: pd.DataFrame, savename=None):
    df = df.copy()
    df.columns = [f"\\thead{{{col}}}" for col in df.columns]
    df.index.names = [f"\\thead{{{idx}}}" if idx else None for idx in df.index.names]
    sty = df.style
    sty = sty.format(precision=2, na_rep="-")
    latex = sty.to_latex(hrules=True)
    if savename:
        with open(savename, "w") as f:
            f.write(latex)
    return latex
