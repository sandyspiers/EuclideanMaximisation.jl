module Generator

#= 

Module used to generate Euclidean Max-Sum problem instances.
To generate each instance, a user may provide the needed parameters, 
generate a random instance, or read an instance from file.

Includes generators for the following problem types:
    - max-sum diversity problem
    - capacitated diversity problem
    - bi-level diveristy problem
    - generalized diversity problem

=#

import Random

using JuMP
using ..Model: EmsModel, build_edm

export random_capacitated_diversity_problem
export file_capacitated_diversity_problem

export random_generalized_diversity_problem
export file_generalized_diversity_problem

export random_bilevel_diversity_problem
export file_bilevel_diversity_problem

export random_diversity_problem
export file_diversity_problem

#= Creates a capacitated diversity problem EmsModel =#
function capacitated_diversity_problem(
    weights::Vector{T} where {T},
    capacity,
    edm::Matrix{T} where {T},
    name = nothing,
)::EmsModel
    # asserations
    n, n2 = size(edm)
    @assert n == n2 == length(weights)
    @assert 0 < capacity < sum(weights)
    # create model
    mdl = JuMP.Model()
    x = @variable(mdl, 0 <= x[1:n] <= 1, Bin)
    # add capacity constraint
    @constraint(mdl, weights' * x <= capacity)
    # return ems
    if !isnothing(name)
        return EmsModel(; mdl = mdl, loc_dvars = x, edm = edm, name = name)
    end
    return EmsModel(; mdl = mdl, loc_dvars = x, edm = edm)
end

#= Randomizes a capacitated diversity problem =#
function random_capacitated_diversity_problem(
    num_locations::Int,
    num_coords::Int,
    capacity_ratio::AbstractFloat,
    seed = nothing,
)::EmsModel
    @assert 0.0 < capacity_ratio < 1.0
    @assert num_locations > 1
    @assert num_coords > 1
    # name it
    name = "rcdp_n$(num_locations)_s$(num_coords)_b$(round(Int, capacity_ratio * 100))"
    # set seed
    if !isnothing(seed)
        Random.seed!(seed)
        name *= "_k$seed"
    end
    # weights and capacity
    weights = rand(0:1000, num_locations)
    capacity = sum(weights) * capacity_ratio
    # Create a distance matrix of random locations
    locations = rand(0:100, (num_locations, num_coords))
    edm = build_edm(locations)
    return capacitated_diversity_problem(weights, capacity, edm, name)
end

#= Randomizes a capacitated diversity problem =#
function rcdp(n::Int, s::Int, b::AbstractFloat, k = nothing)::EmsModel
    return random_capacitated_diversity_problem(n, s, b, k)
end

#= Reads a capacitated diversity problem in the form of an MDPLIB instance =#
function file_capacitated_diversity_problem(filename)
    # get name
    name = split(basename(filename), ".")[1]
    # read lines and write parser
    lns = readlines(filename)
    parse_line(i) = [parse(Float64, x) for x in split(lns[i])]
    # preamble info
    num_locations = parse(Int, lns[1])
    capacity = parse(Int, lns[3])
    weights = parse_line(5)
    # read in edm
    edm = zeros(num_locations, num_locations)
    for i in 1:num_locations
        edm[i, :] .= parse_line(i + 6)
    end
    return capacitated_diversity_problem(weights, capacity, edm, name)
end

#= Reads a capacitated diversity problem in the form of an MDPLIB instance =#
function fcdp(filename)
    return file_capacitated_diversity_problem(filename)
end

#= A variable-cost generalized diveristy problem =#
function generalized_diversity_problem(
    edm::Matrix{T} where {T},
    a::Vector{T} where {T},
    b::Vector{T} where {T},
    c::Vector{T} where {T},
    demand,
    budget,
    name = nothing,
)
    # assertions
    n, n2 = size(edm)
    @assert n == n2 == length(a) == length(b) == length(c)

    mdl = JuMP.Model()

    @variable(mdl, 0 <= x[1:n] <= 1, Bin)
    @variable(mdl, 0 <= t[1:n], Int)

    @constraint(mdl, sum(t) >= demand)
    @constraint(mdl, a' * x + b' * t <= budget)
    @constraint(mdl, [i = 1:n], t[i] <= c[i] * x[i])

    # return ems
    if !isnothing(name)
        return EmsModel(; mdl = mdl, loc_dvars = x, edm = edm, name = name)
    end
    return EmsModel(; mdl = mdl, loc_dvars = x, edm = edm)
end

#= randomizes a generalized diveristy problem =#
function random_generalized_diversity_problem(
    num_locations,
    num_coords,
    demand_ratio,
    cost_ratio,
    seed = nothing,
)
    # name it
    name = "rgdp_n$(num_locations)_s$(num_coords)_b$(round(Int, demand_ratio * 100))_k$(round(Int, cost_ratio * 100))"
    # set seed
    if !isnothing(seed)
        Random.seed!(seed)
        name *= "_seed$seed"
    end
    # generate instance parameters
    c = rand(1:1000, num_locations)
    a = c ./ 2 .+ rand(num_locations) .* (2 .* c .- c ./ 2) # uniform between c_i/2,2c_i
    b = (min.(a, 1) + rand(num_locations) .* (max.(a, 1) - min.(a, 1))) ./ 100
    # Create a distance matrix of random locations
    locations = rand(0:100, (num_locations, num_coords))
    edm = build_edm(locations)

    return generalized_diversity_problem(
        edm,
        a,
        b,
        c,
        demand_ratio * sum(c),
        cost_ratio * (sum(a) + b' * c),
        name,
    )
end

#= randomizes a generalized diveristy problem =#
function rgdp(
    num_locations,
    num_coords,
    demand_ratio,
    cost_ratio,
    seed = nothing,
)
    return random_generalized_diversity_problem(
        num_locations,
        num_coords,
        demand_ratio,
        cost_ratio,
        seed,
    )
end

#= reads a generalized diversity problem in the format of an MDPLIB file (also reads demand and budget) =#
function file_generalized_diversity_problem(filename)
    name = split(basename(filename), ".")[1]
    # read lines and preamble
    lns = readlines(filename)
    n = parse(Int, lns[1])
    # read edm
    edm = zeros(n, n)
    k = 2
    for i in 1:n-1
        for j in i+1:n
            d = parse(Float64, split(lns[k])[end])
            k += 1
            edm[i, j] = d
            edm[j, i] = d
        end
    end
    # read variable cost info
    a = zeros(n)
    b = zeros(n)
    c = zeros(n)
    for i in 1:n
        a[i], b[i], c[i] = parse.(Float64, split(lns[k])[2:end])
        k += 1
    end
    # get budget and demand from file
    b1, b2, demand = parse.(Int, split(lns[k]))
    budget = b1 + b2

    return generalized_diversity_problem(edm, a, b, c, demand, budget, name)
end

#= reads a generalized diversity problem in the format of an MDPLIB file (also reads demand and budget) =#
function fgdp(filename)
    return file_generalized_diversity_problem(filename)
end

#= creates a EmsModel bilevel diversity problem =#
function bilevel_diverisity_problem(
    edm::Matrix{T} where {T},
    cardinality,
    threshold,
    name = nothing,
)
    # assertions
    n, n2 = size(edm)
    @assert n == n2
    @assert 0 < cardinality < n
    @assert threshold < maximum(edm)
    # create model
    mdl = JuMP.Model()
    x = @variable(mdl, 0 <= x[1:n] <= 1, Bin)
    # add cardinality constraint
    @constraint(mdl, sum(x) == cardinality)
    # add conflicts
    for i in 1:n-1
        for j in i+1:n
            if edm[i, j] < threshold
                @constraint(mdl, x[i] + x[j] <= 1)
            end
        end
    end
    # return ems
    if !isnothing(name)
        return EmsModel(; mdl = mdl, loc_dvars = x, edm = edm, name = name)
    end
    return EmsModel(; mdl = mdl, loc_dvars = x, edm = edm)
end

#= randomizes a bilevel diversity problem. threshold is taken as a percentage between 0 and max(d[i,j]) =#
function random_bilevel_diversity_problem(
    num_locations::Int,
    num_coords::Int,
    capacity_ratio::AbstractFloat,
    threshold,
    seed = nothing,
)::EmsModel
    @assert 0.0 < capacity_ratio < 1.0
    @assert threshold <= one(threshold)

    # name it
    name = "rcdp_n$(num_locations)_s$(num_coords)_b$(round(Int, capacity_ratio * 100))_rho$(round(Int, threshold * 100))"
    # set seed
    if !isnothing(seed)
        Random.seed!(seed)
        name *= "_k$seed"
    end
    # determine cardinality
    cardinality = floor(num_locations * capacity_ratio)
    # Create a distance matrix of random locations
    locations = rand(0:100, (num_locations, num_coords))
    edm = build_edm(locations)
    @assert size(edm) == (num_locations, num_locations)
    # threshold as proportion of maximum distance
    threshold = threshold * maximum(edm)
    return bilevel_diverisity_problem(edm, cardinality, threshold, name)
end

#= randomizes a bilevel diversity problem =#
function rbdp(n::Int, s::Int, b::AbstractFloat, rho, k = nothing)::EmsModel
    return random_bilevel_diversity_problem(n, s, b, rho, k)
end

#= reads a bilevel diversity problem in the MDPLIB form, with a given threshold =#
function file_bilevel_diversity_problem(filename, capacity_ratio, threshold)
    @assert 0.0 < capacity_ratio < 1.0
    @assert threshold <= one(threshold)
    # Get info from filename
    test_name =
        split(basename(filename), ".")[1] *
        "_b$(round(Int, capacity_ratio * 100))_rho$(round(Int, threshold * 100))"

    lns = readlines(filename)
    n = parse(Int, split(lns[1])[1])

    edm = zeros(n, n)
    k = 2
    for i in 1:n-1
        for j in i+1:n
            d = parse(Float64, split(lns[k])[end])
            k += 1
            edm[i, j] = d
            edm[j, i] = d
        end
    end
    cardinality = floor(n * capacity_ratio)
    # threshold as proportion of maximum distance
    threshold = threshold * maximum(edm)
    return bilevel_diverisity_problem(edm, cardinality, threshold, test_name)
end

function fbdp(filename, capacity_ratio, threshold)
    return file_bilevel_diversity_problem(filename, capacity_ratio, threshold)
end

function random_diversity_problem(
    num_locations::Int,
    num_coords::Int,
    capacity_ratio::AbstractFloat,
    seed = nothing,
)::EmsModel
    return random_bilevel_diversity_problem(
        num_locations,
        num_coords,
        capacity_ratio,
        0,
        seed,
    )
end

function rdp(n::Int, s::Int, b::AbstractFloat, k = nothing)::EmsModel
    return random_diversity_problem(n, s, b, k)
end

function file_diversity_problem(filename, capacity_ratio)
    return file_bilevel_diversity_problem(filename, capacity_ratio, 0)
end

function fdp(filename, capacity_ratio)
    return file_diversity_problem(filename, capacity_ratio)
end

end
