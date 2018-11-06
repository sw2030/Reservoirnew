module Reservoir

include("Stencil.jl")
include("reservoirfunc.jl")
include("res3d.jl")
include("stencilgmres.jl")
include("preconditioning.jl")
include("distributed.jl")

export makegrid, Reservoirmodel, stencilgmres, getresidual, getstencil, ReservoirSolve

end
