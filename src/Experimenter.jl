module Experimenter

#=
Module used to conduct large scale numerical tests of the optimisation algorithms.
The general procedure it to write a setup yaml file, read it here
Multiple experiments can be defined in a single yaml file
This setup file is the produced to get each individual setup needed
We then use Distributed to conduct these experiments
=#

using Distributed: @everywhere, rmprocs, addprocs, pmap, CachingPool
using ProgressMeter: @showprogress
using OrderedCollections: OrderedDict
using Glob: glob

import YAML

import JuMP
import GLPK
import SCIP
import CPLEX
import Gurobi

using ..Solvers: solve!, Result
import ..Generator

export run_experiment

function run_experiment(filename::String)
    experiment_yaml = read_yaml_file(filename)

    if !(experiment_yaml isa Vector)
        experiment_yaml = [experiment_yaml]
    end

    for setup in experiment_yaml
        # start by freeing up some memory
        GC.gc()

        if !get(setup, "run", false)
            continue
        end

        name = setup["name"]
        mkpath("data/setup")
        YAML.write_file("data/setup/$name.yml", setup)

        # Refreash all workers, and get them to import Emsca
        num_workers = get(setup, "workers", 16)
        workers = addprocs(
            num_workers;
            exeflags = "--project=$(Base.active_project())",
        )
        @everywhere workers begin
            Main.eval(quote
                import EuclideanMaximisation
            end)
        end

        # run
        experiments = get_experimental_list(setup)

        # warmstart
        cached_workers = CachingPool(workers)
        @showprogress desc = "warmstart:$name" pmap(
            standard_run_gc,
            cached_workers,
            experiments[1:num_workers],
        )

        # actual run
        results = @showprogress desc = name pmap(
            standard_run_gc,
            cached_workers,
            experiments,
        )
        rmprocs(workers)

        mkpath("data/results")
        write_results("data/results/$name.csv", results)
    end

    return
end

function get_experimental_list(setup)
    #= Gets the full list of experiments =#
    instance_parameters = pop!(setup, "instance")
    optimizer_parameters = pop!(setup, "parameters", nothing)
    setups = product_dictionary(setup)
    instances = get_instance_arguments(instance_parameters)
    experiments = [
        OrderedDict(
            "setup" => s,
            "instance" => i,
            "parameters" => optimizer_parameters,
        ) for s in setups for i in instances
    ]
    return experiments
end

function product_dictionary(dict::OrderedDict{String,Any})
    #= Given a dictionary with lists, 
    returns a list of dictionaries of all permutations of elements in each list =#
    # make strings non-iterable
    for (key, value) in dict
        if typeof(value) <: AbstractString
            dict[key] = (value,)
        end
    end
    # product through values, and zip back into dictionaries
    key = keys(dict)
    prod = Iterators.product(values(dict)...)
    dicts = [OrderedDict(zip(key, p)) for p in prod]
    # return the dictionary
    return dicts
end

function get_instance_arguments(parameters::OrderedDict{String,Any})
    #= Get the list of instance arguments =#
    if haskey(parameters, "filename")
        parameters["filename"] = glob(parameters["filename"])
    end
    # make strings non-iterable
    for (key, value) in parameters
        if typeof(value) <: AbstractString
            parameters[key] = (value,)
        end
    end
    repeats = pop!(parameters, "repeats", 1)
    seed = pop!(parameters, "seed", -1)
    # product through values, and zip back into dictionaries
    key = keys(parameters)
    prod = Iterators.product(values(parameters)...)
    parameters = [OrderedDict(zip(key, p)) for p in prod for i in 1:repeats]
    if seed > 0
        for para in parameters
            para["seed"] = seed
            seed += 1
        end
    end
    return parameters
end

function generate_instance(dict)
    # create generating function
    instance_generator = getfield(Generator, Symbol(dict["setup"]["generator"]))
    return instance_generator(values(dict["instance"])...)
end

function standard_run_gc(experiment_dict)
    # remove stdout (progressmeter goes to stderr)
    oldstdout = stdout
    Base.redirect_stdout()

    res = standard_run(experiment_dict)

    # release memory
    GC.gc()

    # retrieve stdout
    Base.redirect_stdout(oldstdout)

    return res
end

function standard_run(experiment_dict)
    # setup
    setup = experiment_dict["setup"]

    # Generate instance
    ems = generate_instance(experiment_dict)

    # Set the optimizer
    JuMP.set_optimizer(ems.mdl, get_optimizer_by_name(setup["optimizer"]))

    # set time_limit
    JuMP.set_time_limit_sec(ems.mdl, setup["timelimit"])

    # set thread limit
    try
        JuMP.set_attribute(
            ems.mdl,
            JuMP.MOI.NumberOfThreads(),
            setup["optimizer_threads"],
        )
    catch
    end

    # Dont display output
    JuMP.set_silent(ems.mdl)

    # set parameters
    if !isnothing(experiment_dict["parameters"])
        if haskey(experiment_dict["parameters"], setup["optimizer"])
            for (key, value) in
                experiment_dict["parameters"][setup["optimizer"]]
                try
                    JuMP.set_attribute(ems.mdl, key, value)
                catch
                end
            end
        end
    end

    result = solve!(ems, setup["solver"])
    res = OrderedDict(
        string(key) => getfield(result, key) for key in fieldnames(Result)
    )
    result = OrderedDict(experiment_dict..., "result" => res)

    # flatten
    pop!(result, "parameters")
    result = flatten_name(result, "setup")
    result = flatten_name(result, "instance")
    result = flatten_name(result, "result")
    return result
end

function read_yaml_file(filename::String)
    return YAML.load_file(filename; dicttype = OrderedDict{String,Any})
end

function comma_concat(xs...)
    l = ""
    for x in xs[1:end-1]
        l *= string(x) * ","
    end
    return l * string(xs[end]) * "\n"
end

function flatten_name(dict, name)
    #= flattens multi-level dictionary =#
    _dict = pop!(dict, name)
    return OrderedDict(dict..., _dict...)
end

function write_results(
    filename::String,
    results::Vector{OrderedDict{String,Any}},
)
    file = open(filename, "w")
    write(file, comma_concat(keys(results[1])...))
    for res in results
        write(file, comma_concat(values(res)...))
    end
    close(file)
    return
end

function get_optimizer_by_name(optimizer_name)
    optimizer_name = lowercase(optimizer_name)

    if contains(optimizer_name, "cplex")
        return () -> CPLEX.Optimizer()
    end
    if contains(optimizer_name, "glpk")
        return () -> GLPK.Optimizer()
    end
    if contains(optimizer_name, "gurobi")
        return () -> Gurobi.Optimizer()
    end
    if contains(optimizer_name, "scip")
        return () -> SCIP.Optimizer()
    end

    @error "$optimizer_name is not a valid optimizer"
end

end
