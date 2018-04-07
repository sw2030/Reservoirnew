function make_P_E{TS}(S::Stencil{TS,3,7})
    nx, ny, nz = size(S)
    parray = Array{eltype(S.v),3}(nx, ny, nz)
    earray = Array{eltype(S.v),3}(nx, ny, nz)
    SMzero = @SMatrix(zeros(2,2))
    for i in 1:nx, j in 1:ny, k in 1:nz
        earray[i,j,k] = StencilPoint{StaticArrays.SArray{Tuple{2,2},Float64,2,4},3,7}(Base.setindex(S[i,j,k].value, @SMatrix(zeros(2,2)),4))
        parray[i,j,k] = StencilPoint{StaticArrays.SArray{Tuple{2,2},Float64,2,4},3,7}((SMzero,SMzero,SMzero,inv(S[i,j,k].value[4]),SMzero,SMzero,SMzero))
    end
    Pinv = Stencil{StaticArrays.SArray{Tuple{2,2},Float64,2,4},3,7,typeof(parray)}(parray)
    E    = Stencil{StaticArrays.SArray{Tuple{2,2},Float64,2,4},3,7,typeof(parray)}(earray)

    return Pinv, E
end
function precond_1{TS,Tx}(Pinv::Stencil{TS,3,7},E::Stencil{TS,3,7},x::Grid{Tx,3,7})
    return Pinv*(x-E*(Pinv*x))
end
