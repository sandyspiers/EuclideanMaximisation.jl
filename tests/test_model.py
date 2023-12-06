import numpy as np
import pytest

from emsca.model import EmsModel

""" 
Tests model constructors on example instances of the randomized capacitated diversity problem
"""


def test_location_based_constructor():
    m = EmsModel()
    x = m.binary_var_list(10)
    for _x in x:
        m.set_dvar_location(_x, np.random.randint(0, 100, size=2))
    w = np.random.randint(0, 100, size=len(x))
    m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())

    m.check_model()


def test_set_edm_constructor():
    m = EmsModel()
    x = m.binary_var_list(10)

    locs = np.random.randint(0, 100, size=(len(x), 2))
    magnitudes = np.tile(np.sum(np.square(locs), axis=1), reps=(locs.shape[0], 1))
    gram = locs.dot(locs.T)
    edm = np.sqrt(np.maximum(magnitudes + magnitudes.T - 2 * gram, 0))
    m.set_edm(edm)

    w = np.random.randint(0, 100, size=len(x))
    m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())

    m.check_model()


def test_failure_no_location_dvars():
    with pytest.raises(Exception):
        m = EmsModel()
        x = m.binary_var_list(10)
        w = np.random.randint(0, 100, size=len(x))
        m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())
        m.check_model()


def test_failure_adding_negative_dvars():
    with pytest.raises(Exception):
        m = EmsModel()
        x = m.continuous_var_list(10, lb=-10)
        for _x in x:
            m.set_dvar_location(_x, np.random.randint(0, 100, size=2))
        w = np.random.randint(0, 100, size=len(x))
        m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())
        m.maximize(1)
        m.check_model()


def test_failure_adding_objective():
    with pytest.raises(Exception):
        m = EmsModel()
        x = m.binary_var_list(10)
        for _x in x:
            m.set_dvar_location(_x, np.random.randint(0, 100, size=2))
        w = np.random.randint(0, 100, size=len(x))
        m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())
        m.maximize(x[0])
        m.check_model()


def test_failure_mismatched_set_size():
    with pytest.raises(Exception):
        m = EmsModel()
        x = m.binary_var_list(10)

        locs = np.random.randint(0, 100, size=(len(x), 2))
        magnitudes = np.tile(np.sum(np.square(locs), axis=1), reps=(locs.shape[0], 1))
        gram = locs.dot(locs.T)
        edm = np.sqrt(np.maximum(magnitudes + magnitudes.T - 2 * gram, 0))
        m.set_edm(edm, [x[0]])

        w = np.random.randint(0, 100, size=len(x))
        m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())

        m.check_model()


def test_failure_noneuclidean():
    with pytest.raises(Exception):
        m = EmsModel()
        x = m.binary_var_list(10)

        edm = np.random.randint(0, 25, size=(len(x), len(x)))
        edm += edm.T
        edm[np.diag_indices_from(edm)] = 0
        m.set_edm(edm)

        w = np.random.randint(0, 100, size=len(x))
        m.add_constraint(m.dot(x, w) <= 0.5 * w.sum())

        m.check_model()
