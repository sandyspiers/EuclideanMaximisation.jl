import numpy as np

from emsca.model import EmsModel

m = EmsModel()
x = m.binary_var_list(1000)
for _x in x:
    m.set_dvar_location(_x, np.random.randint(0, 100, size=20))
w = np.random.randint(0, 100, size=len(x))
m.add_constraint(m.sum(x) <= 10)
m.verbose = 3
m.add_lp_tangents = True
m.parameters.timelimit = 20
m.solve()
