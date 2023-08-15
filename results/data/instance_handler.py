import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


import numpy as np
from emsca.model import EmsModel


def random_capacitated_diversity_problem(
    num_locations, num_coords, capacity_ratio, seed=None
):
    """
    Initialise a random capacitated diversity problem, and returns the EmsModel.
    """
    assert capacity_ratio > 0
    assert capacity_ratio < 1
    assert num_coords > 1
    assert num_locations > 1
    if seed is not None:
        np.random.seed(seed)

    m = EmsModel(
        name=f"rcdp_n{num_locations}_s{num_coords}_b{int(capacity_ratio*100)}_seed{seed}"
    )
    x = m.binary_var_list(num_locations)
    for _x in x:
        m.set_dvar_location(_x, np.random.randint(0, 100, size=num_coords))
    w = np.random.randint(0, 1000, size=num_locations)
    m.add_constraint(m.dot(x, w) <= capacity_ratio * w.sum())
    return m


def capacitated_diversity_problem_from_file(filename):
    """
    Returns a capacitated diversity problem from file.
    """
    # Get the capacity ratio from filename
    test_name = filename.split("/")[-1]
    capacity_ratio = int(test_name.split("b")[-1][:2])
    assert capacity_ratio > 0
    assert capacity_ratio < 100

    # Turn the filename into test name
    test_name = test_name.split(".")[0]

    with open(filename, "r") as f:
        lns = f.readlines()
        np_line = lambda i: np.array([float(x) for x in lns[i].split()])
        n = int(lns[0])
        c = int(lns[2])
        w = np_line(4)
        edm = np.zeros((n, n))
        for i in range(n):
            edm[i] = np_line(i + 6)
    assert n > 0
    assert c > 0
    m = EmsModel(f"cdp_{test_name}_{capacity_ratio}")
    x = m.binary_var_list(n)
    m.add_constraint(m.dot(x, w) <= c)
    m.set_edm(edm)
    return m


def random_generalized_diversity_problem(
    num_locations, num_coords, capacity_ratio, cost_ratio, seed=None
):
    """
    Returns a ranodmized generalized diversity problem instance from file.
    Options:
     - capacity_ratio = [0.2,0.3]
     - cost_ratio = [0.5,0.6]
    """
    assert num_locations > 1
    assert num_coords > 1
    assert capacity_ratio > 0
    assert capacity_ratio < 1
    assert cost_ratio > 0
    assert cost_ratio < 1
    if seed is not None:
        np.random.seed(seed)

    # Generate instance parameters
    c = np.random.randint(1, 1000, size=num_locations)
    a = np.random.rand(num_locations)
    a = c / 2 + np.multiply(a, (2 * c - c / 2))  # uniform between c_i/2,2c_i
    b = np.random.rand(num_locations)
    b = (np.minimum(a, 1) + np.multiply(b, np.maximum(a, 1) - np.minimum(a, 1))) / 100

    # Generate random locations
    locations = np.random.randint(1, 100, size=(num_locations, num_coords))
    magnitudes = np.tile(
        np.sum(np.square(locations), axis=1), reps=(locations.shape[0], 1)
    )
    gram = locations.dot(locations.T)
    edm = np.maximum(magnitudes + magnitudes.T - 2 * gram, 0)
    edm = np.sqrt(edm)

    # Create model
    m = EmsModel(
        f"rgdp_n{num_locations}_s{num_coords}_b{int(capacity_ratio*100)}_k{int(cost_ratio*100)}_seed{seed}"
    )
    x = m.binary_var_list(num_locations, name="x")
    m.set_edm(edm, x)
    B = c.sum() * capacity_ratio
    K = (a.sum() + b.dot(c)) * cost_ratio
    t = m.integer_var_list(num_locations, name="t")
    m.add_constraint(m.sum(t) >= B)
    m.add_constraint(m.dot(x, a) + m.dot(t, b) <= K)
    m.add_constraints(t[i] <= c[i] * x[i] for i in range(num_locations))
    return m


def generalized_diversity_problem_from_file(filename):
    """
    Returns a generalized diversity problem instance from file.
    """
    # Get info from filename's
    test_name = filename.split("/")[-1]
    capacity_ratio = int(test_name.split("b")[-1][:2])
    cost_ratio = int(test_name.split("k")[-1][:2])
    assert capacity_ratio > 0
    assert capacity_ratio < 10
    assert cost_ratio > 0
    assert cost_ratio < 10
    test_name = test_name.split("/")[-1].split(".")[0]
    with open(filename, "r") as f:
        lns = f.readlines()
        n = int(lns[0])
        edm = np.zeros((n, n))
        a = np.zeros(n)
        b = np.zeros(n)
        c = np.zeros(n)
        for i in range(1, int(n * (n - 1) / 2) + 1):
            ii, jj, d = lns[i].split()
            ii = int(ii) - 1
            jj = int(jj) - 1
            edm[ii, jj] = float(d)
            edm[jj, ii] = float(d)
        for j in range(i + 1, i + n + 1):
            _l = lns[j].split()
            _i = int(_l[0]) - 1
            a[_i] = float(_l[1])
            b[_i] = float(_l[2])
            c[_i] = float(_l[3])
        _l = lns[j + 1].split()

    m = EmsModel(f"gdp_{test_name}_b{capacity_ratio}_k{cost_ratio}")
    x = m.binary_var_list(n, name="x")
    m.set_edm(edm)
    B = c.sum() * capacity_ratio * 0.1
    K = (a.sum() + b.dot(c)) * cost_ratio * 0.1
    t = m.integer_var_list(n, name="t")
    m.add_constraint(m.sum(t) >= B)
    m.add_constraint(m.dot(x, a) + m.dot(t, b) <= K)
    m.add_constraints(t[i] <= c[i] * x[i] for i in range(n))

    return m
