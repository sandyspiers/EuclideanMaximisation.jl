import numpy as np

from emsca.model import EmsModel

""" 
Tests the 4 solvers with all available settings. 
Makes sure they work, and are exact 
"""


def test_glover():
    m = EmsModel()
    x = m.binary_var_list(10)
    for _x in x:
        m.set_dvar_location(_x, np.random.randint(0, 100, size=2))
    w = np.random.randint(0, 100, size=len(x))
    m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())
    m.solve(solver="glover")
    m.report_solve()


def test_quad_no_lp_tangents():
    m = EmsModel()
    x = m.binary_var_list(10)
    for _x in x:
        m.set_dvar_location(_x, np.random.randint(0, 100, size=2))
    w = np.random.randint(0, 100, size=len(x))
    m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())
    m.add_lp_tangents = False
    m.solve(solver="quad")
    m.report_solve()


def test_quad_rootonly_lp_tangents():
    m = EmsModel()
    x = m.binary_var_list(10)
    for _x in x:
        m.set_dvar_location(_x, np.random.randint(0, 100, size=2))
    w = np.random.randint(0, 100, size=len(x))
    m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())
    m.add_lp_tangents = "rootonly"
    m.solve(solver="quad")
    m.report_solve()


def test_quad_all_lp_tangents():
    m = EmsModel()
    x = m.binary_var_list(10)
    for _x in x:
        m.set_dvar_location(_x, np.random.randint(0, 100, size=2))
    w = np.random.randint(0, 100, size=len(x))
    m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())
    m.add_lp_tangents = True
    m.solve(solver="quad")
    m.report_solve()


def test_repoa_no_lp_tangents():
    m = EmsModel()
    x = m.binary_var_list(10)
    for _x in x:
        m.set_dvar_location(_x, np.random.randint(0, 100, size=2))
    w = np.random.randint(0, 100, size=len(x))
    m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())
    m.add_lp_tangents = False
    m.solve(solver="repoa")
    m.report_solve()


def test_repoa_rootonly_lp_tangents():
    m = EmsModel()
    x = m.binary_var_list(10)
    for _x in x:
        m.set_dvar_location(_x, np.random.randint(0, 100, size=2))
    w = np.random.randint(0, 100, size=len(x))
    m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())
    m.add_lp_tangents = "rootonly"
    m.solve(solver="repoa")
    m.report_solve()


def test_repoa_all_lp_tangents():
    m = EmsModel()
    x = m.binary_var_list(10)
    for _x in x:
        m.set_dvar_location(_x, np.random.randint(0, 100, size=2))
    w = np.random.randint(0, 100, size=len(x))
    m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())
    m.add_lp_tangents = True
    m.solve(solver="repoa")
    m.report_solve()


def test_glover_quad_agree():
    seed = np.random.randint(1, 1000)
    solvers = ["glover", "quad"]
    obj_vals = {}
    for s in solvers:
        np.random.seed(seed)
        m = EmsModel()
        x = m.binary_var_list(10)
        for _x in x:
            m.set_dvar_location(_x, np.random.randint(0, 100, size=2))
        w = np.random.randint(0, 100, size=len(x))
        m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())
        m.solve(solver=s)
        obj_vals[s] = m.objective_value
    assert abs(obj_vals["glover"] - obj_vals["quad"]) <= EmsModel.REL_TOL


def test_glover_repoa_no_lp_tangents_agree():
    seed = np.random.randint(1, 1000)
    solvers = ["glover", "repoa"]
    obj_vals = {}
    for s in solvers:
        np.random.seed(seed)
        m = EmsModel()
        x = m.binary_var_list(10)
        for _x in x:
            m.set_dvar_location(_x, np.random.randint(0, 100, size=2))
        w = np.random.randint(0, 100, size=len(x))
        m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())
        m.add_lp_tangents = False
        m.solve(solver=s)
        obj_vals[s] = m.objective_value
    assert abs(obj_vals["glover"] - obj_vals["repoa"]) <= EmsModel.REL_TOL


def test_glover_repoa_root_only_lp_tangents_agree():
    seed = np.random.randint(1, 1000)
    solvers = ["glover", "repoa"]
    obj_vals = {}
    for s in solvers:
        np.random.seed(seed)
        m = EmsModel()
        x = m.binary_var_list(10)
        for _x in x:
            m.set_dvar_location(_x, np.random.randint(0, 100, size=2))
        w = np.random.randint(0, 100, size=len(x))
        m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())
        m.add_lp_tangents = "rootonly"
        m.solve(solver=s)
        obj_vals[s] = m.objective_value
    assert abs(obj_vals["glover"] - obj_vals["repoa"]) <= EmsModel.REL_TOL


def test_glover_repoa_all_lp_tangents_agree():
    seed = np.random.randint(1, 1000)
    solvers = ["glover", "repoa"]
    obj_vals = {}
    for s in solvers:
        np.random.seed(seed)
        m = EmsModel()
        x = m.binary_var_list(10)
        for _x in x:
            m.set_dvar_location(_x, np.random.randint(0, 100, size=2))
        w = np.random.randint(0, 100, size=len(x))
        m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())
        m.add_lp_tangents = True
        m.solve(solver=s)
        obj_vals[s] = m.objective_value
    assert abs(obj_vals["glover"] - obj_vals["repoa"]) <= EmsModel.REL_TOL


def test_glover_fcard_no_lp_tangents_agree():
    seed = np.random.randint(1, 1000)
    solvers = ["glover", "fcard"]
    obj_vals = {}
    for s in solvers:
        np.random.seed(seed)
        m = EmsModel()
        x = m.binary_var_list(10)
        for _x in x:
            m.set_dvar_location(_x, np.random.randint(0, 100, size=2))
        w = np.random.randint(0, 100, size=len(x))
        m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())
        m.add_lp_tangents = False
        m.solve(solver=s)
        obj_vals[s] = m.objective_value
    assert abs(obj_vals["glover"] - obj_vals["fcard"]) <= EmsModel.REL_TOL


def test_glover_fcard_root_only_lp_tangents_agree():
    seed = np.random.randint(1, 1000)
    solvers = ["glover", "fcard"]
    obj_vals = {}
    for s in solvers:
        np.random.seed(seed)
        m = EmsModel()
        x = m.binary_var_list(10)
        for _x in x:
            m.set_dvar_location(_x, np.random.randint(0, 100, size=2))
        w = np.random.randint(0, 100, size=len(x))
        m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())
        m.add_lp_tangents = "rootonly"
        m.solve(solver=s)
        obj_vals[s] = m.objective_value
    assert abs(obj_vals["glover"] - obj_vals["fcard"]) <= EmsModel.REL_TOL


def test_glover_fcard_all_lp_tangents_agree():
    seed = np.random.randint(1, 1000)
    solvers = ["glover", "fcard"]
    obj_vals = {}
    for s in solvers:
        np.random.seed(seed)
        m = EmsModel()
        x = m.binary_var_list(10)
        for _x in x:
            m.set_dvar_location(_x, np.random.randint(0, 100, size=2))
        w = np.random.randint(0, 100, size=len(x))
        m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())
        m.add_lp_tangents = True
        m.solve(solver=s)
        obj_vals[s] = m.objective_value
    assert abs(obj_vals["glover"] - obj_vals["fcard"]) <= EmsModel.REL_TOL
