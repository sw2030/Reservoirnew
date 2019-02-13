using CuArrays, LinearAlgebra,  NVTX
using CUDAdrv

include("helper.jl")
using .myprofile

myprofile.foo(2)
myprofile.foo(2)
NVTX.@activate CUDAdrv.@profile CuArrays.@sync myprofile.foo(2)

	
