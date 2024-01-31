# Load Revise.jl
try
    using Revise
catch e
    @warn "Error initializing Revise"
end

# Load OhMyREPL.jl
try
    using OhMyREPL
catch e
    @warn "Error initializing OhMyREPL"
end
