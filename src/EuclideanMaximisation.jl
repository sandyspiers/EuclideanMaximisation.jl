module EuclideanMaximisation

# Import EmsModel, build_edm and other model utilities
include("Model.jl")
using .Model
export Model

# Import solver routines
include("Solvers.jl")
using .Solvers
export Solvers

# Instance generators
include("Generator.jl")
using .Generator
export Generator

# Experiment handels
include("Experimenter.jl")
using .Experimenter
export Experimenter

end
