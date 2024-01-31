module Model

import Distances
import LinearAlgebra
import JuMP

using Parameters: @with_kw

export EmsModel, build_edm

const GRAMM_PSD_TOLERANCE = -1e-6

#= 
The base struct for a Euclidean Max-Sum Model.
Includes a JuMP model, a list of location-associated decision variables, and a Euclidean distance matrix
=#
@with_kw struct EmsModel
    mdl::JuMP.Model
    loc_dvars::Vector{JuMP.VariableRef}
    edm::Matrix{TypeNum} where {TypeNum<:Real}
    name::String = "ems_model"
end

# Build an edm base on set of locations
# each row is a location
function build_edm(locations, locations_by_row = true)
    if locations_by_row
        return Distances.pairwise(Distances.Euclidean(), locations; dims = 1)
    end
    return Distances.pairwise(Distances.Euclidean(), locations; dims = 2)
end

# Check a distance matrix is square & euclidean
function check_edm_valid(edm::Matrix{T})::Bool where {T<:Real}
    # check square
    n, n2 = size(edm)
    if n != n2
        @warn "EDM is not square!"
        return false
    end
    # construct a simplified grammian
    magnitudes = edm[1, :] * ones(T, n)'
    gramm = magnitudes + magnitudes' - edm

    # check psd
    if !LinearAlgebra.issymmetric(gramm)
        @warn "Grammian is not symmetric!"
        return false
    elseif minimum(LinearAlgebra.eigvals(gramm)) < GRAMM_PSD_TOLERANCE
        @warn "Grammian is not PSD! Min eva: $(minimum(LinearAlgebra.eigvals(gramm)))"
        return false
    end

    return true
end

# Check model is valid for Euclidean-Max-Sum-Cutting algorithm
function check_model_valid(ems::EmsModel)::Bool
    # check that ems.mdl has no objecitve
    if JuMP.objective_function(ems.mdl) != 0
        @warn "Provided model had objective function, however this will be ignored."
    end

    # check locations dvars are nonnegative
    for var in ems.loc_dvars
        if JuMP.lower_bound(var) < 0
            @warn "Model includes negative location dvars!"
            return false
        end
    end

    #  check that distance matrix is in euclidean
    if !check_edm_valid(ems.edm)
        @warn "Distance matrix is not euclidean!"
    end

    return true
end

#= return a deepcopy of an EmsModel =#
function copy(ems::EmsModel)::EmsModel
    mdl = JuMP.copy(ems.mdl)
    x = ems.loc_dvars
    loc_dvars = mdl[:x]
    edm = Base.copy(ems.edm)
    return EmsModel(; mdl = mdl, loc_dvars = loc_dvars, edm = edm)
end

end
