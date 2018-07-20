function make_P_E_precond_1(MS::MStencil{4,Float64,3,7})
    nx, ny, nz = size(MS)
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

#=
    parray = Array{eltype(S.v),3}(nx, ny, nz) # inverse of S[4] values
    earray = Array{eltype(S.v),3}(nx, ny, nz) # other things
    SMzero = @SMatrix(zeros(2,2))
    for i in 1:nx, j in 1:ny, k in 1:nz
        earray[i,j,k] = StencilPoint{StaticArrays.SArray{Tuple{2,2},Float64,2,4},3,7}(Base.setindex(S[i,j,k].value, @SMatrix(zeros(2,2)),4))
        parray[i,j,k] = StencilPoint{StaticArrays.SArray{Tuple{2,2},Float64,2,4},3,7}((SMzero,SMzero,SMzero,inv(S[i,j,k].value[4]),SMzero,SMzero,SMzero))
    end
    Pinv = Stencil{StaticArrays.SArray{Tuple{2,2},Float64,2,4},3,7,typeof(parray)}(parray)
    E    = Stencil{StaticArrays.SArray{Tuple{2,2},Float64,2,4},3,7,typeof(parray)}(earray)
=#

    return Pinv, E
end
function precond_1{TS,Tx}(Pinv::MStencil{4,TS,3,7},E::MStencil{4,TS,3,7},x::MGrid{2,Tx,3,7})
    return Pinv*(x-E*(Pinv*x))
end
function make_P_E_precond_2{TS}(S::Stencil{TS,3,7})
    nx, ny, nz = size(S)
    parray = Array{eltype(S.v),3}(nx, ny, nz)
    earray = Array{eltype(S.v),3}(nx, ny, nz)
    SMzero = @SMatrix(zeros(2,2))
    for i in 1:nx, j in 1:ny, k in 1:nz
        tmp_e = Base.setindex(S[i,j,k].value,@SMatrix(zeros(2,2)),4)
        tmp_e = Base.setindex(tmp_e, @SMatrix(zeros(2,2)),1)
        tmp_e = Base.setindex(tmp_e, @SMatrix(zeros(2,2)),7)
        earray[i,j,k] = StencilPoint{StaticArrays.SArray{Tuple{2,2},Float64,2,4},3,7}(tmp_e)
        parray[i,j,k] = StencilPoint{StaticArrays.SArray{Tuple{2,2},Float64,2,4},3,7}((inv(S[i,j,k].value[1]),SMzero,SMzero,inv(S[i,j,k].value[4]),SMzero,SMzero,inv(S[i,j,k].value[7])))
    end
    Pinv = Stencil{StaticArrays.SArray{Tuple{2,2},Float64,2,4},3,7,typeof(parray)}(parray)
    E    = Stencil{StaticArrays.SArray{Tuple{2,2},Float64,2,4},3,7,typeof(parray)}(earray)

    return Pinv, E
end
