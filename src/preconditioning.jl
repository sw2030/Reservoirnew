function make_P_E_precond_1(MS::MStencil{4,Float64,3,7})
    nx, ny, nz = size(MS.stencils[1])
    parrays = [copy(MS.stencils[i].v) for i in 1:4]
    earrays = [copy(MS.stencils[i].v) for i in 1:4]
    for k in 1:nz, j in 1:ny, i in 1:nx
        for c in 1:4
            earrays[c][i,j,k] = StencilPoint{Float64,3,7}(Base.setindex(earrays[c][i,j,k].value,0.0,4))
        end
        v1, v2, v3, v4 = inv([parrays[(row-1)*2 + col][i,j,k].value[4] for row in 1:2, col in 1:2])
        parrays[1][i,j,k] = StencilPoint{Float64,3,7}(Base.setindex(Tuple(zeros(7)),v1,4))
        parrays[2][i,j,k] = StencilPoint{Float64,3,7}(Base.setindex(Tuple(zeros(7)),v3,4))
        parrays[3][i,j,k] = StencilPoint{Float64,3,7}(Base.setindex(Tuple(zeros(7)),v2,4))
        parrays[4][i,j,k] = StencilPoint{Float64,3,7}(Base.setindex(Tuple(zeros(7)),v4,4))
    end
    Pinv = MStencil{4,Float64,3,7,typeof(parrays[1])}(Tuple(Stencil{Float64,3,7,typeof(parrays[1])}.(parrays)))
    E    = MStencil{4,Float64,3,7,typeof(parrays[1])}(Tuple(Stencil{Float64,3,7,typeof(parrays[1])}.(earrays)))

    return Pinv, E, parrays, earrays
end
function precond_1(Pinv::MStencil{4,TS,3},E::MStencil{4,TS,3},x::MGrid{2,Tx,3}) where {TS,Tx}    
    result = copy(x)
    tmp = zero(x)
    gemv!(1, Pinv, x, 0, tmp)
    gemv!(-1, E, tmp, 1, result)
    gemv!(1, Pinv, result, 0, tmp)
    return tmp
end
function precond_2(Pinv::MStencil{4,TS,3},E::MStencil{4,TS,3},x::MGrid{2,Tx,3}) where {TS,Tx} 
    tmp1 = copy(x)
    tmp2 = Pinv*x
    gemv!(-1, E, tmp2, 1, tmp1)   # tmp1 = x - EPx
    gemv!(1, Pinv, tmp1, 0, tmp2) # tmp2 = P(x-EPx)
    copyto!(tmp1, x)              # tmp1 = x
    gemv!(-1, E, tmp2, 1, tmp1)   # tmp1 = x - E(P(x-EPx))
    gemv!(1, Pinv, tmp1, 0, tmp2) # tmp2 = P(x-E(P(x-EPx)))
    return tmp2
end
