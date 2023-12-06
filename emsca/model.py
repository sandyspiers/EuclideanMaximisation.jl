import sys
import time

import numpy as np
from cplex.callbacks import LazyConstraintCallback
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin
from docplex.mp.model import Model as CplexModel
from docplex.mp.sdetails import SolveDetails

from .relax_linear import LinearRelaxer


class EmsModel(CplexModel):
    """
    Euclidean Max-Sum Model & Solver for problems of type
    ```
    max  <Qx,x>
    s.t. Ax <= a
         x >= 0
    ```
    where Q is euclidean distance matrix.

    This object inherits a `docplex.mp.model.Model` (which we renamed `CplexModel`).
    As such, models can be created as they might normally be using the `docplex` package.

    The Euclidean maximisation objective cannot be added directely.
    Instead, users can introduced the distance matrix (and hence objective) in 1 of 2 ways.
        1. Set locations.
            - Set locations of each of the associated dvars.
              this is achieved by `set_dvar_location(dvar,location)`.
              Once all locations are added, the user can call `build_edm()`
              which quickly creates the euclidean distance matrix.
              This method is also called from `check_model()`, id the `edm` is `None`.
        2. Set edm
            - Sets the edm using the `set_edm(edm,locations)`.
              The locations must either be `None`, in which case all variables are assumed
              to match the locations, in the order they were created.
              Otherwise, locations must be a list of dvars matching the size of the `edm`
              provided.

    Models must fit the above formulation!
    As such,
     * It cannot handel extra linear terms,
     * All location-associated variables must be non-negative.

    Once create, the model can be solved using 1 of 4 exact solution methods:
     1. Quadratic CPLEX
     2. Glover Linearisation
     3. OA-MIP's
     4. Forced Cardinalities
    See `docs/algorithms.md` for detailed information on each algorithm.
    """

    # Global solver settings
    REL_TOL = 1e-4
    LP_TANGENT_ITER_REL_TOL = 1e-4
    MAX_UB = 1e99
    LP_TANGENTS_MAX_ITER = 50
    SOLVER_OPTIONS = ["repoa", "fcard", "quad", "glover", "concave_oa"]
    LP_TANGENT_OPTIONS = [True, False, "rootonly"]
    ALL_SOLVER_OPTIONS = [
        ("repoa", False),
        ("repoa", "rootonly"),
        ("repoa", True),
        ("fcard", False),
        ("fcard", "rootonly"),
        ("fcard", True),
        ("quad", False),
        ("glover", False),
        ("concave_oa", False),
    ]

    # Logger settings
    LOG_LEVELS = {"RESULT": 1, "INFO": 2, "CPLEX": 3}

    def __init__(self, name=None):
        """
        Creates an EmsModel, which is a child of a `docplex.mp` Model.
        That way, users can add constraints and variables as normal.
        """
        CplexModel.__init__(self, name)

        #### Locations and distance matrix ####
        # A dictionary that takes variable(docplex.dvar) -> location(np.ndarry)
        self.locations = {}
        # List of location-associated dvars
        self.loc_dvars = []
        # The full Euclidean distance matrix
        self.edm: np.ndarray = None

        #### Solver Variables ####
        # The lp relaxation, needed for root cuts
        self.relaxed_model: CplexModel = None
        self.relaxed_model.var_map: dict
        self.relaxed_model.ct_map: dict
        # The epigraph variable, needed for outer approximation
        self.epigraph_var = None

        #### Solver settings ####
        # Solver, options including ["repoa" (default), "fcard", "quad", "glover"]
        self.solver = "repoa"
        # Add LP tangents always, never, or only at root
        self.add_lp_tangents = False  # Options: True, False, "rootonly"
        # Use 1 thread by defaul
        self.parameters.threads = 1
        # Verbose level
        self.verbose = 1

        #### Solver Metric ####
        # Some time counters for algorithmic procedures
        self._original_timelimit = 0
        self.iteration_cum_time = 0
        self.lp_tangents_cumtime = 0
        self.max_card_sp_cumtime = 0
        # Cut counters
        self.iterations = 0
        self.lp_cut_counter = 0
        self.ip_cut_counter = 0

    def set_dvar_location(self, dvar, location):
        """Sets the location of dvar and puts into locations dictionary"""
        self.locations[dvar] = location

    def set_edm(self, euclidean_distance_matrix: np.ndarray, loc_dvars: list = None):
        """
        Sets the euclidean distance matrix, and the associated locations variables.
        If no variables are provided, it is assumed all variables will be used.
        """
        self.edm = euclidean_distance_matrix
        if loc_dvars is None:
            self.loc_dvars = [x for x in self.iter_variables()]
        else:
            self.loc_dvars = loc_dvars
        if self.edm.shape[0] != len(self.loc_dvars):
            raise f"Mismatched shape.  Distance matrix is {self.edm.shape} but model has {len(self.loc_dvars)} location variables"

    def build_edm(self):
        """
        Provided the model contains no objective, and the locations of dvars is provided, this
        method will build the distance matrix quickly using the gram matrix and the cosine rule.
        """
        # Get the variables that have associated locations
        self.loc_dvars = list(self.locations.keys())
        # Get the EDM from the gramm matrix & cosine rule
        locs = np.stack(list(self.locations.values()), axis=0)
        magnitudes = np.tile(np.sum(np.square(locs), axis=1), reps=(locs.shape[0], 1))
        gram = locs.dot(locs.T)
        edm = np.maximum(magnitudes + magnitudes.T - 2 * gram, 0)
        self.edm = np.sqrt(edm)

    def check_model(self):
        """
        Checks the validity of the EmsModel, including,
         1. Objective contains no other linear objective
         2. Check distance matrix contains at most 1 positive eigenvalue.
            If contains all negative eigenvalues, solution algorithms will still work, but the user
            is warned that the instances is not a euclidean distance matrix.
         3. Location variables are nonnegative
        """

        # 1. Check there is no objective
        if self.get_objective_expr().number_of_variables() > 0:
            raise Exception("Cannot handle extra objective terms!")

        # 2. Checks distance matrix
        if self.edm is None:
            self.build_edm()
        if len(self.loc_dvars) == 0:
            raise Exception("No locations variables provided!")
        second_largest_eigenvalue = np.linalg.eigvalsh(self.edm)[-2]
        if second_largest_eigenvalue > self.REL_TOL:
            raise Exception(
                f"Distance matrix is not Euclidean! Second largest eigenvalue = {second_largest_eigenvalue}"
            )

        # 3. Assert all location dvars they all nonnegative
        for _x in self.loc_dvars:
            if _x.lb < 0:
                raise Exception(
                    "All location-associated variables must be nonnegative!"
                )

        return True

    def f(self, x):
        """Returns f(x)"""
        return x.dot(self.edm).dot(x) / 2

    def df(self, x):
        """Returns df(x)"""
        return x.dot(self.edm)

    def solve(self, solver=None, add_lp_tangents=None, check_model=False, **kwargs):
        """
        Solve the model using 1 of four methods:
         1. Quadratic Cplex
         2. Glover linearisation w/ Cplex
         3. OA-MIP's
         4. Cardinalities

        Any keyword arguments are passed to Cplex.solve methods
        """
        # Check model
        if check_model:
            self.check_model()
        elif self.edm is None:
            self.build_edm()

        # Log the name and statistics of the model
        if self._if_log("INFO"):
            self.print_information()

        # Check parameters
        self._original_timelimit = self.parameters.timelimit.value
        if solver is not None:
            self.solver = solver
        if add_lp_tangents is not None:
            self.add_lp_tangents = add_lp_tangents
        if self.add_lp_tangents not in self.LP_TANGENT_OPTIONS:
            raise Exception(
                f"Please choose a valid root cut option from {self.LP_TANGENT_OPTIONS}"
            )

        # Send to associated solver method
        if self.solver == "quad":
            sol = self._solve_quad(**kwargs)
        elif self.solver == "glover":
            sol = self._solve_glover(**kwargs)
        elif self.solver == "repoa":
            sol = self._solve_repoa(**kwargs)
        elif self.solver == "fcard":
            sol = self._solve_fcard(**kwargs)
        elif self.solver == "concave_oa":
            sol = self._solve_concave_oa(**kwargs)
        else:
            raise Exception(
                f"Not a valid solution approach! Please choose from {EmsModel.SOLVER_OPTIONS}"
            )

        self.report_solve("RESULT")

        return sol

    def report_solve(self, level=0):
        lr = lambda s: self._log(level, s)
        lr(f"    ==================================")
        lr(f"  | Solved model {self.name}")
        lr(f"  | Contains {len(self.loc_dvars)} location variables")
        lr(f"  | Timelimit    = {self._original_timelimit}")
        lr(f"  | Solver       = {self.solver}")
        lr(f"  | Root-cut lvl = {self.add_lp_tangents}")
        lr(f"  | ----------------------------------")
        lr(f"  | Solvetime  = {self.solve_details.time}")
        lr(f"  | Objective  = {self.objective_value}")
        lr(f"  | Bestbound  = {self.solve_details.best_bound}")
        lr(f"  | ----------------------------------")
        lr(f"  | SP iterations  = {self.iterations}")
        lr(f"  | LP-cuts added  = {self.lp_cut_counter}")
        lr(f"  | IP-cuts added  = {self.ip_cut_counter}")
        lr(f"    ==================================")

    def _if_log(self, level):
        if isinstance(level, str):
            if self.verbose < self.LOG_LEVELS[level]:
                return False
        elif isinstance(level, int):
            if self.verbose < level:
                return False
        return True

    def _log(self, level, *args):
        if self._if_log(level):
            print(*args)

    def _setup_relaxed_model(self):
        """Setup the relaxed model"""
        self.relaxed_model, var_map = LinearRelaxer().linear_relaxation(self)
        self.relaxed_model.loc_dvars = [var_map[i] for i in self.loc_dvars]
        self.relaxed_model.var_map = var_map
        self.relaxed_model.ct_map = {}
        self.relaxed_model.epigraph_var = var_map[self.epigraph_var]

    def _solve_quad(self, **kwargs):
        """Solve the problem using quadratic cplex"""
        objective = 0
        # Add the quadratic objective term
        for i in range(len(self.loc_dvars)):
            for j in range(i + 1, len(self.loc_dvars)):
                objective += self.loc_dvars[i] * self.loc_dvars[j] * self.edm[i, j]
        # solve using standard cplex (the parent of this class)
        self.maximize(objective)
        self._log("INFO", "Solving by quadratic CPLEX...")
        return super().solve(log_output=self._if_log("CPLEX"), **kwargs)

    def _solve_glover(self, **kwargs):
        """Solve the problem using glover linearisation"""
        # Introduce our glover variables
        w = self.continuous_var_list(len(self.loc_dvars) - 1, name="glover_w")
        self.add_constraints(
            w[i] <= self.loc_dvars[i] * self.edm[i, (i + 1) :].sum()
            for i in range(len(self.loc_dvars) - 1)
        )
        self.add_constraints(
            w[i]
            <= self.sum(
                self.loc_dvars[j] * self.edm[i, j]
                for j in range(i + 1, len(self.loc_dvars))
            )
            for i in range(len(self.loc_dvars) - 1)
        )
        # solve using standard cplex (the parent of this class)
        self.maximize(self.sum(w))
        self._log("INFO", "Solving by Glover linearisation...")
        return super().solve(log_output=self._if_log("CPLEX"), **kwargs)

    def _solve_concave_oa(self, **kwargs):
        """
        Solve the problem using outer approximation on the concaved objective
        For detailed description of algorithm see `docs/algorithms.md`
        """
        #### Setup Solver ####
        self.solvetime = time.time()

        # Determine largest eigenvalue
        rho = np.linalg.eigvalsh(self.edm).max()
        # Offset from matrix
        _temp_ub = self.edm.sum()
        self.edm[np.diag_indices_from(self.edm)] -= rho

        # Add epigraph variable
        # No lower bound, as f(x) <= 0 for all x (as f is now made concave)
        self.epigraph_var = self.continuous_var(
            ub=_temp_ub, lb=-1e99, name="epigraph_var"
        )
        # offset by the rho
        self.maximize(self.epigraph_var + self.sum(self.loc_dvars) * rho / 2)

        # Register standard callback
        tangent_planes = self.register_callback(TangentPlanes)
        tangent_planes.ems_model = self

        # Solve
        self._log("INFO", "Solving by Concaved Outer-Approximation")
        return super().solve(log_output=self._if_log("CPLEX"), **kwargs)

    def _solve_repoa(self, **kwargs):
        """
        Solve the problem using repeating OA-MIP's.
        For detailed description of algorithm see `docs/algorithms.md`
        """
        #### Setup Solver ####
        self._log("INFO", "Solving by OA-MIPs")
        self.solvetime = time.time()

        # Add epigraph variable
        self.epigraph_var = self.continuous_var(ub=self.edm.sum(), name="epigraph_var")
        self.maximize(self.epigraph_var)

        # Setup relaxed model
        if self.add_lp_tangents is not False:
            self._setup_relaxed_model()

        # Begin with max cardinality cut
        self._add_max_cardinality_tangent_plane()

        # Add lp tangents
        if self.add_lp_tangents == "rootonly":
            self._add_lp_tangents()

        #### Solver Iterations ####
        self._log("INFO", "Beginning OA-MIP's interations...")
        LB = 0.1
        UB = self.MAX_UB
        while (UB - LB) / LB > self.REL_TOL:
            self.iterations += 1
            self._log("INFO", f"OA-MIP iteration {self.iterations}")

            # Add lp tangents
            if self.add_lp_tangents is True:
                self._add_lp_tangents()

            # Solve the OA-MIP
            self._log("INFO", "Solving OA-MIP subproblem")
            remaining_time = self._original_timelimit - (time.time() - self.solvetime)
            if remaining_time <= 0:
                break
            self.parameters.timelimit = remaining_time
            sol = super().solve(log_output=self._if_log("CPLEX"), **kwargs)
            self.iteration_cum_time += self.solve_details.time

            # Add tangent plane, adding to relaxed model only if add_root_cuts is True
            loc_var_sol = np.array(sol.get_value_list(self.loc_dvars))
            fy = self._add_tangent_plane(
                loc_var_sol, add_to_relaxed_model=self.add_lp_tangents
            )
            self.ip_cut_counter += 1

            # Update bounds
            UB = self.objective_value
            LB = max(fy, LB)
            self._log(
                "INFO", f"Solved OA-MIP subproblem in {self.solve_details.time} seconds"
            )
            self._log("INFO", f"LB: {LB}\tUB:{UB}")

        #### Update solve details ####
        # The update solve_details based on the performance of the repoa algorithm
        sol = self.solution
        sol._objective = [LB]
        self._set_solution(sol)
        self.solvetime = time.time() - self.solvetime
        self._solve_details = SolveDetails(
            time=self.solvetime, status_code=1, best_bound=UB
        )
        # The optimal solution was found in the last iteration, so leave as is

        self._log("INFO", f"Completed OA-MIP's...")
        self._log("INFO", f"\t{self.iteration_cum_time} seconds in OA-MIP")
        self._log("INFO", f"\t{self.lp_tangents_cumtime} seconds in root cuts")
        self._log("INFO", f"\t{self.max_card_sp_cumtime} seconds in max card SP")
        self._log("INFO", f"\t---------------------")
        self._log("INFO", f"\t{self.solvetime} total seconds")
        self._log("INFO", f"\t---------------------")

        return self.solution

    def _solve_fcard(self, **kwargs):
        """
        Solve the problem using decreasing forced cardinality.
        For detailed description of algorithm see `docs/algos.md`
        """
        #### Solver Setup ####
        self._log("INFO", "Solving by decreasing cardinality")
        self.solvetime = time.time()

        # Add epigraph variable
        self.epigraph_var = self.continuous_var(ub=self.edm.sum(), name="epigraph_var")
        self.maximize(self.epigraph_var)

        # Setup relaxed model
        if self.add_lp_tangents is not False:
            self._setup_relaxed_model()

        # Find the max cardinality and add cardinality constraint
        cardinality = self._add_max_cardinality_tangent_plane()
        cardinality_ct = self._add_constraint(
            self.sum(self.loc_dvars) == cardinality,
            add_to_relaxed_model=self.add_lp_tangents is not False,
        )

        # Add lp tangents
        if self.add_lp_tangents == "rootonly":
            self._add_lp_tangents()

        # Register callback
        tangent_planes = self.register_callback(TangentPlanes)
        tangent_planes.ems_model = self

        ##### Solve iterations ####
        self._log("INFO", "Beginning cardinality interations...")
        prev_sol = None
        LB = 0.1
        UB = self.MAX_UB
        while cardinality > 0 and (UB - LB) / LB > self.REL_TOL:
            self.iterations += 1
            self._log(
                "INFO",
                f"Cardinality iteration {self.iterations} with forced cardinality {cardinality}",
            )

            # Add lp tangents
            if self.add_lp_tangents is True:
                self._add_lp_tangents()

            # Solve the forced card subproblem
            self._log("INFO", f"Solving cardinality forced SP")
            remaining_time = self._original_timelimit - (time.time() - self.solvetime)
            if remaining_time <= 0:
                break
            self.parameters.timelimit = remaining_time
            sol = super().solve(log_output=self._if_log("CPLEX"), **kwargs)
            self.iteration_cum_time += self.solve_details.time
            self._log(
                "INFO",
                f"Solved  cardinality forced SP in {self.solve_details.time} seconds",
            )
            self._log("INFO", f"Total cuts added = {tangent_planes.nb_cuts}")

            # Save solution, update lower bound
            if prev_sol is None:
                prev_sol = sol
            prev_sol = sol
            LB = max(LB, prev_sol.get_objective_value())

            # Calculate upper bound
            # Remove cardinality constraint and replace with <=
            self._remove_constraint(
                cardinality_ct, remove_from_relaxed_model=self.add_lp_tangents
            )
            cardinality_ct = self._add_constraint(
                self.sum(self.loc_dvars) <= cardinality,
                add_to_relaxed_model=self.add_lp_tangents,
            )

            # Solve upper bound problem
            self._log("INFO", f"Solving upper bounding problem")
            remaining_time = self._original_timelimit - (time.time() - self.solvetime)
            if remaining_time <= 0:
                break
            self.parameters.timelimit = remaining_time
            sol = super().solve(log_output=self._if_log("CPLEX"), **kwargs)
            self.iteration_cum_time += self.solve_details.time
            UB = max(sol.get_objective_value(), LB)

            # Update cardinality
            self._remove_constraint(
                cardinality_ct, remove_from_relaxed_model=self.add_lp_tangents
            )
            cardinality -= 1
            cardinality_ct = self._add_constraint(
                self.sum(self.loc_dvars) == cardinality,
                add_to_relaxed_model=self.add_lp_tangents,
            )

        #### Update solve details ####
        # Set the solution to the solution from the previous solve
        self._set_solution(prev_sol)

        # Update solve details from previous algorithm
        self.solvetime = time.time() - self.solvetime
        self._solve_details = SolveDetails(
            time=self.solvetime, status_code=1, best_bound=UB
        )

        # Update counter from callback
        self.ip_cut_counter += tangent_planes.nb_cuts

        self._log("INFO", "Completed cardinality forced algo...")
        self._log("INFO", f"\t{self.iteration_cum_time} seconds in card iterations")
        self._log("INFO", f"\t{self.lp_tangents_cumtime} seconds in root cuts")
        self._log("INFO", f"\t{self.max_card_sp_cumtime} seconds in max card SP")
        self._log("INFO", f"\t---------------------")
        self._log("INFO", f"\t{self.solvetime} total seconds")
        self._log("INFO", f"\t---------------------")

        return self.solution

    def _add_max_cardinality_tangent_plane(self):
        """
        Determines maxmimum cardinality and adds a tangent plane of such.
        This is the easiest way to get a valid starting cut
        """
        self._log("INFO", "Solving maximum cardinality subproblem")
        # Previous objective
        prev_objective = self.get_objective_expr()

        # Find the max cardinality
        self.maximize(self.sum(self.loc_dvars))
        sol = super().solve()
        self.max_card_sp_cumtime += self.solve_details.time

        # Get solution and add its tangent planes
        loc_var_sol = np.array(sol.get_value_list(self.loc_dvars))
        self._add_tangent_plane(
            loc_var_sol,
            add_to_relaxed_model=self.add_lp_tangents is not False,
        )
        self.ip_cut_counter += 1

        # Return to previous objective
        self.maximize(prev_objective)

        # Return maximum cardinality
        self._log(
            "INFO",
            f"Solved maximum cardinality subproblem in {self.solve_details.time} seconds",
        )
        self._log("INFO", f"Maximum cardinality = {self.objective_value}")
        return self.objective_value

    def _add_lp_tangents(self):
        """Add the LP-tangent planes"""
        # Get the LP relaxation
        _t = time.time()
        self._log("INFO", "Beggining root-cut iterations")
        UB = self.MAX_UB
        LB = 0.1
        iteration = 0
        while (
            (UB - LB) / LB > self.LP_TANGENT_ITER_REL_TOL
            and iteration <= self.LP_TANGENTS_MAX_ITER
        ):
            # Solve and check termination conditions
            iteration += 1
            sol = self.relaxed_model.solve(log_output=False)
            UB = sol.get_objective_value()
            if sol is None:
                break

            # Get solution and add its tangent planes
            loc_var_sol = np.array(sol.get_value_list(self.relaxed_model.loc_dvars))
            fy = self._add_tangent_plane(loc_var_sol, add_to_relaxed_model=True)
            LB = max(LB, fy)
            self.lp_cut_counter += 1

        _t = time.time() - _t
        self.lp_tangents_cumtime += _t
        self._log(
            "INFO",
            f"Completed root-cut iterations ({_t} secs, {iteration-1} cuts, {iteration} iter)",
        )

    def _quick_f_df(self, x):
        """
        Returns f(x),df(x).
        Calculates these at the same time, to avoid unneccesary calculations.
        """
        df = x.dot(self.edm)
        f = df.dot(x) / 2
        return f, df

    def _add_tangent_plane(self, loc_var_sol, add_to_relaxed_model=False):
        """
        Turns the solution into a tangent plane and adds to model, and maybe the LP.
        Returns the objective value of the solution
        """
        # Get solution and add its tangent planes
        fy, dfy = self._quick_f_df(loc_var_sol)
        self._add_constraint(
            self.epigraph_var
            <= fy - dfy.dot(loc_var_sol) + self.dot(self.loc_dvars, dfy),
            add_to_relaxed_model=add_to_relaxed_model,
        )
        return fy

    def _add_cut_to_relaxed_model(self, cut):
        """
        Adds the cut to the relaxed model.
        WARNING: Does not check if it SHOULD add cut!
        Please check before calling this method.
        """
        copied_ct = cut.relaxed_copy(self.relaxed_model, self.relaxed_model.var_map)
        self.relaxed_model.ct_map[cut] = self.relaxed_model.add(copied_ct)

    def _add_constraint(self, ct, ctname=None, add_to_relaxed_model=False):
        """
        Calls the normal add constraint procedure, but before adding,
        checks to see if the constraint should added to the relaxed model
        """
        if add_to_relaxed_model is True:
            self._add_cut_to_relaxed_model(ct)
        return self.add_constraint(ct, ctname)

    def _remove_constraint(self, ctarg, remove_from_relaxed_model=False):
        """
        Calls the normal remove constraint procedure, but before removing,
        checks to see if the constraint should also be removed from relaxed model
        """
        if remove_from_relaxed_model is True:
            self.relaxed_model.remove_constraint(self.relaxed_model.ct_map[ctarg])
        return self.remove_constraint(ctarg)


class TangentPlanes(ConstraintCallbackMixin, LazyConstraintCallback):
    """
    Lazy constraint callback to add tangent planes to model.
    Once instantiated, must add an attribute 'm' which is the EmsModel.
    """

    def __init__(self, env):
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.nb_cuts = 0

    def __call__(self):
        try:
            self.nb_cuts += 1

            # Shorthanding...
            m: EmsModel = self.ems_model
            x = m.loc_dvars
            theta = m.epigraph_var

            # fetch variable values to make solution of x
            y = np.array(self.make_solution_from_vars(x).get_value_list(x))

            # Calculate cut
            fy, dfy = m._quick_f_df(y)
            cut = theta <= fy - dfy.dot(y) + m.dot(x, dfy)

            # Add to the model, including relaxed model, if required
            if m.add_lp_tangents is True:
                m._add_cut_to_relaxed_model(cut)
            coef, sence, rhs = self.linear_ct_to_cplex(cut)
            self.add(coef, sence, rhs)

        except:
            print(sys.exc_info()[0])
            raise
