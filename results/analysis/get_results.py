import numpy as np
import pandas as pd


def read_results(filepath):
    # Read results.log file
    df = pd.read_csv(
        filepath,
        header=None,
        names=[
            "name",
            "timelimit",
            "solver",
            "lp_tangent",
            "solvetime",
            "obj_val",
            "bound",
            "iterations",
            "LPcuts",
            "IPcuts",
        ],
        dtype={
            "solvetime": float,
            "obj_val": float,
            "bound": float,
            "iterations": int,
            "LPcuts": int,
            "IPcuts": int,
        },
    )

    # Type
    def get_type(n: str):
        if n.startswith("cdp"):
            return "CDP"
        if n.startswith("gdp"):
            return "GDP"
        if n.startswith("rgdp"):
            return "RGDP"
        if n.startswith("rcdp"):
            return "RCDP"
        raise f"Unknown instance type for test {n}"

    df.insert(2, "type", [get_type(n) for n in df["name"]])

    # Num
    def get_num(n: str):
        return int(n.split("n")[1].split("_")[0])

    df.insert(3, "num", [get_num(n) for n in df["name"]])

    # Gap
    df["gap"] = (df["bound"] - df["obj_val"]) / df["obj_val"]

    # Dev
    names = ["_".join(n.split("_")[:-2]) for n in df["name"]]
    max_vals = [df[df["name"].str.startswith(n)]["obj_val"].max() for n in names]
    df["dev"] = (max_vals - df["obj_val"]) / df["obj_val"]

    return df


def get_performance_profiles(df):
    """Returns a performance profile s of all unique solver setups within df"""
    # Get all unique sovers
    solvers = set(zip(df["solver"], df["lp_tangent"]))
    pp = {}
    for s, lpt in solvers:
        times = [0] + list(
            df[
                (df["solver"] == s)
                & (df["lp_tangent"] == lpt)
                & (df["solvetime"] <= df["timelimit"])
            ]["solvetime"].sort_values()
        )
        times.append(df["timelimit"].max())
        steps = np.arange(len(times))
        steps[-1] -= 1
        pp[s, lpt] = (times, steps)
    return pp


if __name__ == "__main__":
    df = read_results("results/results.log")
    print(df.head())
