""" 

Reads the results from data/results
and produces the tables and figures as in the paper

"""

import pathlib

import matplotlib.pyplot as plt
import pandas as pd

from utils import read_df, fix_col_names, load_yaml, to_latex, glob_stack

# List of names of cut plane solvers
cut_planes = ["repoa", "fcard", "fcard50", "fcard100"]


def num_cut_boxplot(style_guide):
    """Box plot of number of each type of cut"""
    dfs = [
        "data/results/rcdp-1-thread.csv",
        "data/results/fcdp-1-thread.csv",
        "data/results/rgdp-1-thread.csv",
        "data/results/fgdp-1-thread.csv",
    ]
    df = pd.concat(read_df(df) for df in dfs)
    df = df[df["solved"]]

    fig, axes = plt.subplots(1, 2, sharey=True, width_ratios=[4, 2])

    axes[0].boxplot(
        [df[df["solver"] == sol]["num_cuts"] for sol in cut_planes],
        labels=[style_guide[sol]["short_name"] for sol in cut_planes],
        widths=0.45,
    )
    lp_cuters = ["fcard50", "fcard100"]
    axes[1].boxplot(
        [df[df["solver"] == sol]["num_lp_cuts"] for sol in lp_cuters],
        labels=[style_guide[sol]["short_name"] for sol in lp_cuters],
        widths=0.45,
    )

    axes[0].annotate("Forced Cardinality", (0.33, 0.035), xycoords="figure fraction")
    axes[1].set_xlabel("Forced Cardinality")

    axes[0].set_title("Integer Cuts Added")
    axes[1].set_title("LP Cuts Added")

    fig.suptitle("Number of Cuts Added")

    plt.tight_layout()

    pathlib.Path("analysis/figs/").mkdir(parents=True, exist_ok=True)
    plt.savefig("analysis/figs/cut_boxplot.pdf")


def card_runtime_plot(style_guide):
    """How cardinality affects runtime"""
    dfs = [
        "data/results/rcdp-1-thread.csv",
        "data/results/fcdp-1-thread.csv",
        "data/results/rgdp-1-thread.csv",
        "data/results/fgdp-1-thread.csv",
    ]
    df = pd.concat(read_df(df) for df in dfs)
    df["card_ratio"] = df["max_cardinality"] - df["sol_cardinality"]

    fig, axes = plt.subplots(1)
    for solver in cut_planes:
        if solver == "fcard50":
            continue
        gdf = df[df["solver"] == solver].groupby("card_ratio")["run_time"]

        ln = axes.plot(gdf.mean(), label=style_guide[solver]["label"])[0]
        axes.fill_between(
            gdf.min().index,
            gdf.quantile(0.25),
            gdf.quantile(0.75),
            color=ln._color,
            alpha=0.2,
        )

    axes.set_xlim(left=0)
    axes.set_ylim(bottom=0)
    axes.legend()

    axes.set_title("Effect of cardinality on run time")
    axes.set_ylabel("Solve time (sec)")
    axes.set_xlabel(
        "Difference between maximum cardinality and cardinality of optimal solution"
    )

    pathlib.Path("analysis/figs/").mkdir(parents=True, exist_ok=True)
    plt.savefig("analysis/figs/card_runtime.pdf")


def get_rho_table(style_guide):
    df = read_df("data/results/fbdp-d-1-thread.csv")
    pt = df.pivot_table("run_time", ["name", "n", "rho"], "solver", "mean")
    pt = fix_col_names(pt, style_guide)
    pt = pt[pt.columns[:4]]
    pt[[n >= 250 for _, n, _ in pt.index]]
    return pt


def get_gkd_c_table(style_guide):
    """Some metrics of performance on GKD-c"""
    df = read_df("data/results/fdp-c-1-thread.csv")

    def one_row_table(metric, func):
        tab = df.pivot_table(metric, "name", "solver", func, dropna=False)
        tab = fix_col_names(tab, style_guide)
        tab.index = [metric]
        return tab

    metric_func = [
        ("gap", "mean"),
        ("obj_value", "mean"),
        ("num_cuts", "mean"),
        ("solved", "sum"),
    ]

    return pd.concat([one_row_table(m, f) for m, f in metric_func])


if __name__ == "__main__":
    print("Overall Summary DF")
    style_guide = load_yaml("analysis/style_guide.yml")
    print(
        to_latex(
            glob_stack("data/results/*", style_guide), "analysis/tab/all_table.tex"
        )
    )

    print("\n\nBi-Level")
    print(to_latex(get_rho_table(style_guide), "analysis/tab/bi_level_table.tex"))

    print("\n\nGKD-c")
    print(to_latex(get_gkd_c_table(style_guide), "analysis/tab/gkd-c_table.tex"))

    print("\n\nBoxplot of Num Cuts")
    num_cut_boxplot(style_guide)

    print("\n\nCard v. Runtime Plot")
    card_runtime_plot(style_guide)
