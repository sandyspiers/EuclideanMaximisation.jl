module Solvers

import JuMP
import GLPK
import SCIP
import CPLEX

using ..Model: EmsModel, check_model_valid

export AVALIABLE_SOLVERS, Result, solve!

AVALIABLE_SOLVERS = ["repoa", "fcard", "fcard50", "fcard100", "quad", "glov"]

const REL_TOL = 1e-6
const CUT_REL_TOL = 1e-6
const LP_REL_TOL = 1e-6
const LP_MAX_ITER = 100

struct Result
    obj_value::AbstractFloat
    best_bound::AbstractFloat
    run_time::AbstractFloat
    num_cuts::Int
    num_iterations::Int
    num_lp_cuts::Int
    max_cardinality::Int
    sol_cardinality::Int
end

function Result(obj_value, best_bound, run_time)
    return Result(obj_value, best_bound, run_time, 0, 0, 0, 0, 0)
end

function Result(
    obj,
    bb,
    rt,
    nc,
    ni,
    nlc,
    max_card::AbstractFloat,
    sol_card::AbstractFloat,
)
    return Result(
        obj,
        bb,
        rt,
        nc,
        ni,
        nlc,
        round(Int, max_card),
        round(Int, sol_card),
    )
end

function solve!(ems::EmsModel, method::String = "repoa")::Result
    # Check model is valid for Emsca methods
    if !check_model_valid(ems)
        return
    end

    # if no solver attached use CPLEX
    if JuMP.solver_name(ems.mdl) == "No optimizer attached."
        JuMP.set_optimizer(ems.mdl, () -> CPLEX.Optimizer())
    end

    # set gap
    JuMP.set_attribute(ems.mdl, JuMP.MOI.RelativeGapTolerance(), REL_TOL)

    method = lowercase(method)
    if contains(method, "repoa")
        return solve_repoa!(ems)
    elseif contains(method, "fcard")
        trust_region = parse_last_digits(method)
        return solve_fcard!(ems, trust_region)
    elseif contains(method, "quad")
        return solve_quad!(ems)
    elseif contains(method, "glov")
        return solve_glov!(ems)
    end
    @error "Not a valid solving method!"
    return
end

function parse_last_digits(str::AbstractString)
    digits = match(r"\d+$", str)
    if !isnothing(digits)
        return parse(Int, digits.match)
    end
end

function remaining_time(start_time, time_limit)
    if isnothing(time_limit)
        return nothing
    end
    return max(0, time_limit - (time() - start_time))
end

#= Solve by repeated outer approximation =#
function solve_repoa!(ems::EmsModel)::Result
    # begin start time
    start_time = time()
    time_limit = JuMP.time_limit_sec(ems.mdl)

    # add epi
    epi = JuMP.@variable(ems.mdl, epi >= 0)

    # generate first tangent by maximising cardinality
    JuMP.@objective(ems.mdl, Max, sum(ems.loc_dvars))
    JuMP.optimize!(ems.mdl)

    # if no solution, problem is infeasible
    if !JuMP.has_values(ems.mdl)
        return Result(-Inf, Inf, time() - start_time)
    end

    # add max cardinality tangent, and save its lower bound
    lb, max_cardinality = add_tangent!(ems, epi)
    sol_cardinality = max_cardinality
    ub = Inf
    ncuts = 1

    # return objective
    JuMP.@objective(ems.mdl, Max, epi)
    while (ub - lb) / lb > REL_TOL
        # update timelimit
        JuMP.set_time_limit_sec(ems.mdl, remaining_time(start_time, time_limit))
        JuMP.optimize!(ems.mdl)
        # if no solution, break
        if !JuMP.has_values(ems.mdl)
            break
        end
        ub = JuMP.objective_value(ems.mdl)
        _lb, _card = add_tangent!(ems, epi)
        if _lb > lb
            lb = _lb
            sol_cardinality = _card
        end
        ncuts += 1
    end

    # return result
    return Result(
        lb,
        ub,
        time() - start_time,
        ncuts,
        ncuts,
        0,
        max_cardinality,
        sol_cardinality,
    )
end

function add_tangent!(ems::EmsModel, epi::JuMP.VariableRef)
    y = JuMP.value.(ems.loc_dvars)
    dfy = ems.edm * y
    fy = y' * dfy / 2
    JuMP.@constraint(ems.mdl, epi <= -fy + dfy' * ems.loc_dvars)
    return fy, sum(y)
end

#= Solve by forced cardinality method =#
function solve_fcard!(ems::EmsModel, trust_level = 0)::Result
    if !isnothing(trust_level)
        @assert 0 <= trust_level <= 100
    end

    # begin start time
    start_time = time()
    # read the time limit from the JuMP model
    time_limit = JuMP.time_limit_sec(ems.mdl)

    # add epi
    epi = JuMP.@variable(ems.mdl, epi >= 0)

    # generate first tangent by maximising cardinality
    JuMP.@objective(ems.mdl, Max, sum(ems.loc_dvars))
    JuMP.optimize!(ems.mdl)

    # if no solution, problem is infeasible
    if !JuMP.has_values(ems.mdl)
        return Result(-Inf, Inf, time() - start_time)
    end

    # add max cardinality tangent, and save its lower bound
    incumbent = JuMP.value.(ems.loc_dvars)
    lb, cardinality = add_tangent!(ems, epi)
    ub = Inf
    ncuts = 1
    iter = 0
    lpcuts = 0
    max_cardinality = cardinality

    # reset objective
    JuMP.@objective(ems.mdl, Max, epi)

    # create tangent callback
    cut_set = []
    _tangent_callback(cb_data) = tangent_callback(ems, epi, cut_set, cb_data)

    # begin iterations
    while (ub - lb) / lb > REL_TOL && cardinality > 0
        iter += 1
        # update cardinality constraint
        cardinality_cts =
            JuMP.@constraint(ems.mdl, sum(ems.loc_dvars) == cardinality)

        # add in an LP cut
        lp_cut_set = add_lp_tangent!(
            ems,
            epi,
            start_time,
            time_limit,
            incumbent,
            trust_level,
        )

        # register callback
        JuMP.set_attribute(
            ems.mdl,
            JuMP.MOI.LazyConstraintCallback(),
            _tangent_callback,
        )

        # update timelimit
        JuMP.set_time_limit_sec(ems.mdl, remaining_time(start_time, time_limit))

        # solve with callback
        JuMP.optimize!(ems.mdl)

        # if no solution, break
        if !JuMP.has_values(ems.mdl)
            break
        end

        # get fn value
        y = JuMP.value.(ems.loc_dvars)
        fy = y' * ems.edm * y / 2
        if fy > lb
            lb = fy
            incumbent = y
        end

        # add back all cuts
        ncuts += length(cut_set)
        JuMP.add_constraint.(ems.mdl, cut_set)
        empty!(cut_set)

        # remove callback
        JuMP.set_attribute(ems.mdl, JuMP.MOI.LazyConstraintCallback(), nothing)

        # remove lp cuts
        lpcuts += length(lp_cut_set)
        JuMP.delete.(ems.mdl, lp_cut_set)

        # update cardinality constraint
        JuMP.delete(ems.mdl, cardinality_cts)
        cardinality_cts =
            JuMP.@constraint(ems.mdl, sum(ems.loc_dvars) <= cardinality)

        # update timelimit
        JuMP.set_time_limit_sec(ems.mdl, remaining_time(start_time, time_limit))

        # solve without callback
        JuMP.optimize!(ems.mdl)

        # update upper bound
        if JuMP.has_values(ems.mdl)
            ub = JuMP.objective_value(ems.mdl)
        end

        # reduce cardinality 
        cardinality -= one(cardinality)

        # remove cardinality constraint
        JuMP.delete(ems.mdl, cardinality_cts)
    end

    return Result(
        lb,
        max(lb, ub),
        time() - start_time,
        ncuts,
        iter,
        lpcuts,
        max_cardinality,
        sum(incumbent),
    )
end

function tangent_callback(
    ems::EmsModel,
    epi::JuMP.VariableRef,
    cut_set::Vector{T} where {T},
    callback_data,
)
    if JuMP.callback_node_status(callback_data, ems.mdl) !=
       JuMP.MOI.CALLBACK_NODE_STATUS_INTEGER
        return
    end
    y = JuMP.callback_value.(callback_data, ems.loc_dvars)
    dfy = ems.edm * y
    fy = y' * dfy / 2
    epi_val = JuMP.callback_value(callback_data, epi)
    if epi_val > fy + CUT_REL_TOL
        cut = JuMP.@build_constraint(epi <= -fy + dfy' * ems.loc_dvars)
        push!(cut_set, cut)
        JuMP.MOI.submit(ems.mdl, JuMP.MOI.LazyConstraint(callback_data), cut)
    end
end

function add_lp_tangent!(
    ems::EmsModel,
    epi,
    start_time,
    time_limit,
    incumbent = nothing,
    trust_level = nothing,
)
    if isnothing(trust_level)
        return []
    end

    undo_relaxation = JuMP.relax_integrality(ems.mdl)
    lb = 0.01
    ub = Inf

    if !isnothing(incumbent)
        n = length(ems.loc_dvars)
        trust_region = JuMP.@constraint(
            ems.mdl,
            sum(ems.loc_dvars[i] for i in 1:n if incumbent[i] == 0.0) +
            sum(1 - ems.loc_dvars[i] for i in 1:n if incumbent[i] == 1.0) <=
            trust_level / 100 * n
        )
    else
        trust_region = nothing
    end

    iter = 1
    lp_cut_set = []
    while (ub - lb) / lb > REL_TOL && iter <= LP_MAX_ITER
        iter += 1
        # update timelimit
        JuMP.set_time_limit_sec(ems.mdl, remaining_time(start_time, time_limit))
        JuMP.optimize!(ems.mdl)
        # if no solution, break
        if !JuMP.has_values(ems.mdl)
            break
        end

        y = JuMP.value.(ems.loc_dvars)
        dfy = ems.edm * y
        fy = y' * dfy / 2

        ub = JuMP.objective_value(ems.mdl)
        lb = max(lb, fy)

        push!(
            lp_cut_set,
            JuMP.@constraint(ems.mdl, epi <= -fy + dfy' * ems.loc_dvars)
        )
    end
    if !isnothing(trust_region)
        JuMP.delete(ems.mdl, trust_region)
    end
    undo_relaxation()
    return lp_cut_set
end

#= Solve in primal quadratic form =#
function solve_quad!(ems::EmsModel)::Result
    # Attached objecitve
    JuMP.@objective(ems.mdl, Max, ems.loc_dvars' * ems.edm * ems.loc_dvars / 2)

    # Solve
    JuMP.optimize!(ems.mdl)

    # return result
    if JuMP.has_values(ems.mdl)
        return Result(
            JuMP.objective_value(ems.mdl),
            JuMP.objective_bound(ems.mdl),
            JuMP.solve_time(ems.mdl),
        )
    end
    return Result(-Inf, Inf, JuMP.solve_time(ems.mdl))
end

#= Solve by Glover linearsation =#
function solve_glov!(ems::EmsModel)::Result
    # introduce glover variables
    n = size(ems.edm)[1]
    w = JuMP.@variable(ems.mdl, w[1:n-1] >= 0)

    # formulate glover linearisation
    JuMP.@constraint(
        ems.mdl,
        [i = 1:n-1],
        w[i] <= ems.loc_dvars[i] * sum(ems.edm[i+1:end, i])
    )
    JuMP.@constraint(
        ems.mdl,
        [i = 1:n-1],
        w[i] <= ems.edm[i+1:end, i]' * ems.loc_dvars[i+1:end]
    )

    # add glover objective
    JuMP.@objective(ems.mdl, Max, sum(w))

    # Solve
    JuMP.optimize!(ems.mdl)

    # return result
    if JuMP.has_values(ems.mdl)
        return Result(
            JuMP.objective_value(ems.mdl),
            JuMP.objective_bound(ems.mdl),
            JuMP.solve_time(ems.mdl),
        )
    end
    return Result(-Inf, Inf, JuMP.solve_time(ems.mdl))
end

end
