module Reservoir

include("Stencil.jl")
include("reservoirfunc.jl")
include("res3d.jl")
include("distributed.jl")
include("gpu.jl")
include("stencilgmres.jl")
include("preconditioning.jl")

export makegrid, Reservoirmodel, stencilgmres, getresidual, getstencil, ReservoirSolve, SPE10Solve

end
