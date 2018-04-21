function make_P_E_precond_1{TS}(S::Stencil{TS,3,7})
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
#=function make_P_E_precond_2{TS}(S::Stencil{TS,3,7})
    nx, ny, nz = size(S)
    parray = Array{eltype(S.v),3}(nx, ny, nz)
    earray = Array{eltype(S.v),3}(nx, ny, nz)
    SMzero = @SMatrix(zeros(2,2))
    tmp_e = S[1,1,1].value
    for i in 1:nx, j in 1:ny, k in 1:nz
        tmp_e = Base.setindex(S[i,j,k].value,@SMatrix(zeros(2,2)),4)
        tmp_e = Base.setindex(tmp_2, @SMatrix(zeros(2,2),5)
        tmp_e = Base.setindex(tmp_2, @SMatrix(zeros(2,2),3)
        earray[i,j,k] = StencilPoint{StaticArrays.SArray{Tuple{2,2},Float64,2,4},3,7}(tmp_e)
        parray[i,j,k] = StencilPoint{StaticArrays.SArray{Tuple{2,2},Float64,2,4},3,7}
                                ((SMzero,SMzero,inv(S[i,j,k].value[3]),inv(S[i,j,k].value[4]),inv(S[i,j,k].value[5]),SMzero,SMzero))
    end
    Pinv = Stencil{StaticArrays.SArray{Tuple{2,2},Float64,2,4},3,7,typeof(parray)}(parray)
    E    = Stencil{StaticArrays.SArray{Tuple{2,2},Float64,2,4},3,7,typeof(parray)}(earray)

    return Pinv, E
end=#
