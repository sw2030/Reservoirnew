import Base.zero
import Base.+
import Base.-
import Base.*
import Base./

immutable StencilPointIterator{N,P}
        idx::NTuple{P,NTuple{N,Int}}
end
immutable StencilPoint{T,N,P}
        value::NTuple{P,T}
end
immutable Sindx{N,P} end

(::Sindx{1,3})() = ((-1,), (0,), (1,))
(::Sindx{2,5})() = ((-1,0),(0,-1),(0,0),(0,1),(1,0))
(::Sindx{3,7})() = ((-1,0,0),(0,-1,0),(0,0,-1),(0,0,0),(0,0,1),(0,1,0),(1,0,0))

Base.start(S::StencilPoint)                               = 1
@inbounds Base.next{T,N,P}(S::StencilPoint{T,N,P}, state) = ((Sindx{N,P}()()[state], S.value[state]), state + 1)#((Sindx{N,P}()()[state], S.value[state]), state + 1)
Base.done{T,N,P}(S::StencilPoint{T,N,P}, state)           = state > P

type Stencil{T,N,P,A}
    v::A #Array{StencilPoint{T,N,P},N}
end

Base.copy{T,N,P,A}(S::Stencil{T,N,P,A})    = Stencil{T,N,P,A}(copy(S.v))
Base.getindex{T}(S::Stencil{T,1}, i)       = S.v[i]
Base.getindex{T}(S::Stencil{T,2}, i, j)    = S.v[i, j]
Base.getindex{T}(S::Stencil{T,3}, i, j, k) = S.v[i, j, k]
Base.eltype{T,N,P}(::Type{StencilPoint{T,N,P}}) = T

type Grid{T,N,P,S<:AbstractArray}<:AbstractArray{T,N}
    A::S
end
#Base.getindex{T,P}(g::Grid{T,2,P}, i, j) = g.A[i + (P >> 2), j + (P >> 2)]
#Base.getindex{T,P}(g::Grid{T,3,P}, i, j) = g.A[i + (P >> 2), j + (P >> 2), k+(P>>2)]
Base.getindex{T}(g::Grid{T,1,3}, i) = g.A[i+1]
Base.getindex{T}(g::Grid{T,2,5}, i, j) = g.A[i+1,j+1]
Base.getindex{T}(g::Grid{T,2,1}, i, j) = g.A[i,j]
Base.getindex{T}(g::Grid{T,3,7}, i, j, k) = g.A[i+1,j+1,k+1]
Base.getindex{T}(g::Grid{T,3,1}, i, j, k) = g.A[i,j,k]
function Base.setindex!{T}(g::Grid{T,1,3}, a, i)
    g.A[i+1] = a
end
function Base.setindex!{T}(g::Grid{T,2,5}, a, i, j)
    g.A[i+1, j+1] = a
end
function Base.setindex!{T}(g::Grid{T,3,7}, a, i, j, k)
    #g.A[i+1, j+1,k+1] = a
    setindex!(g.A, a, i+1, j+1, k+1)
end


Base.norm{T}(g::Grid{T,2,5}) = Base.LinAlg.vecnorm2(g.A[2:end-1, 2:end-1])
Base.norm{T}(g::Grid{T,3,7}) = Base.LinAlg.vecnorm2(g.A[2:end-1, 2:end-1, 2:end-1])
Base.dot{T}(x::Grid{T,2,5}, y::Grid{T,2,5}) = vecdot(x.A[2:end-1, 2:end-1], y.A[2:end-1, 2:end-1])
Base.dot{T}(x::Grid{T,3,7}, y::Grid{T,3,7}) = vecdot(x.A[2:end-1, 2:end-1, 2:end-1], y.A[2:end-1, 2:end-1, 2:end-1])

function gridsize{T,N}(g::Grid{T,N})
    if     N==1 return length(g[1])*length(g)
    elseif N==2 return length(g[1,1])*size(g,1)*size(g,2)
    elseif N==3 return length(g[1,1,1])*size(g,1)*size(g,2)*size(g,3)
    else throw(ArgumentError("N is wrong"))
    end
end


zero{T,N,P}(x::Grid{T,N,P})              = Grid{T,N,P,typeof(x.A)}(zero(x.A))
+{T,N,P}(x::Grid{T,N,P}, y::Grid{T,N,P}) = Grid{T,N,P,typeof(x.A)}(x.A+y.A)
-{T,N,P}(x::Grid{T,N,P}, y::Grid{T,N,P}) = Grid{T,N,P,typeof(x.A)}(x.A-y.A)
-{T,N,P}(x::Grid{T,N,P})                 = Grid{T,N,P,typeof(x.A)}(-x.A)
*{T,N,P}(x::Grid{T,N,P}, y::Float64)     = Grid{T,N,P,typeof(x.A)}(x.A*y)
*{T,N,P}(y::Float64, x::Grid{T,N,P})     = Grid{T,N,P,typeof(x.A)}(x.A*y)
/{T,N,P}(x::Grid{T,N,P}, y::Float64)     = Grid{T,N,P,typeof(x.A)}(x.A/y)

Base.copy!{T,N,P,S}(g1::Grid{T,N,P,S}, g2::Grid{T,N,P,S}) = copy!(g1.A, g2.A)
Base.size{T}(g::Grid{T,1,3}) = (length(g.A)-2, )
Base.size{T}(g::Grid{T,2,1}) = size(g.A)
Base.size{T}(g::Grid{T,2,5}) = (size(g.A, 1)-2, size(g.A, 2)-2)
Base.size{T}(g::Grid{T,3,1}) = size(g.A)
Base.size{T}(g::Grid{T,3,7}) = (size(g.A, 1)-2, size(g.A, 2)-2, size(g.A, 3)-2)
Base.size{T,N,P}(S::Stencil{T,N,P}) = size(S.v)

function makegrid{T}(x::Array{T,1},P)
    r = Grid{T,1,P}(zeros(eltype(x), length(x)+2))
    r.A[2:end-1] = x
    r
end
function makegrid{T}(x::Array{T,2},P)
    r = Grid{T,2,P,Array{T,2}}(zeros(eltype(x),size(x,1)+2,size(x,2)+2))
    r.A[2:end-1, 2:end-1] = x
    r
end
function makegrid{T}(x::Array{T,3},P)
    r = Grid{T,3,P,Array{T,3}}(zeros(eltype(x),size(x,1)+2,size(x,2)+2,size(x,3)+2))
    r.A[2:end-1, 2:end-1, 2:end-1] = x
    r
end
function Base.A_mul_B!{Txy,TS}(y::Grid{Txy,1}, S::Stencil{TS,1}, x::Grid{Txy,1})
    for i in eachindex(y)
        Si = S[i]
        yi = zero(Txy)
        for (idx, value) in Si
            (ix, ) = idx
            yi    += value*x[i+ix]
        end
        y[i] = yi
    end
    return y
end
function Base.A_mul_B!{Txy,TS}(a::Number, S::Stencil{TS,2}, x::Grid{Txy,2},  b::Number, y::Grid{Txy,2})
    scale!(y.A, b)
    s1, s2 = size(y)
    for i = 1:s1, j = 1:s2
        tmp = zero(Txy)
        @inbounds for (idx, value) in S[i,j]
            ix, jy = idx
            tmp   +=  value*x[i+ix, j+jy]
        end
        y[i,j] = a*tmp
    end
    return y
end
function Base.A_mul_B!(a::Number, S::Stencil{TS,3,P}, x::Grid{Txy,3,P},  b::Number, y::Grid{Txy,3,P}) where {Txy,TS,P}
    scale!(y.A, b)
    s1, s2, s3 = size(S.v)
    @inbounds for i = 1:s1, j = 1:s2, k = 1:s3
        tmp = zero(Txy)
         for (idx, value) in S[i,j,k]
            ix, jy, kz = idx
            tmp += value*x[i+ix, j+jy, k+kz]
        end
        y[i,j,k] += a*tmp
    end
    return y
end
#=function Base.A_mul_B!(a::Number, S::Stencil{TS,3,P}, x::Grid{Txy,3,P},  b::Number, y::Grid{Txy,3,P}) where {Txy,TS,P}
    scale!(y.A, b)
    s1, s2, s3 = size(y)
    for i = 1:s1, j = 1:s2, k = 1:s3
        tmp = zero(Txy)
        @inbounds for (idx, value) in S[i,j,k]
            ix, jy, kz = idx
            tmp += value*x[i+ix, j+jy, k+kz]
        end
        y[i,j,k] += a*tmp
    end
    return y
end=#
Base.A_mul_B!{Txy,TS,N,P}(y::Grid{Txy,N,P}, S::Stencil{TS,N,P}, x::Grid{Txy,N,P}) = A_mul_B!(1.0, S, x, 0.0, y)
Base.:*{TS,Tx,N,P,A}(S::Stencil{TS,N,P}, x::Grid{Tx,N,P,A}) = A_mul_B!(Grid{Tx,N,P,A}(zero(x.A)), S, x)

function Base.LinAlg.axpy!{T,N,P,S}(a, D1::Grid{T,N,P,S}, D2::Grid{T,N,P,S})
    LinAlg.axpy!(a, D1.A, D2.A)
    return D2
end
function Base.LinAlg.scale!(D::Grid, a::Number)
    LinAlg.scale!(D.A, a)
    return D
end
#=function fullm{TS}(S::Stencil{TS,2,5})
    nx, ny = size(S)
    SS = spzeros(nx*ny, nx*ny)
    for i in 1:nx, j in 1:ny
        nd = (i-1)*ny+j
        Sv = S.v[i,j]
        SS[nd,nd] += Sv.value[3]
        if i!=1  SS[nd,nd-ny] = Sv.value[1] end
        if i!=nx SS[nd,nd+ny] = Sv.value[5] end
        if j!=1  SS[nd,nd-1]  = Sv.value[2] end
        if j!=ny SS[nd,nd+1]  = Sv.value[4] end
    end
    return SS
end
function fullm{TS}(S::Stencil{TS,3,7})
    nx, ny, nz = size(S)
    SS = spzeros(nx*ny*nz, nx*ny*nz)
    for i in 1:nx, j in 1:ny, k in 1:nz
        nd = (i-1)*ny*nz+(j-1)*nz+k
        Sv = S.v[i,j,k]
        SS[nd,nd] += Sv.value[5]
        if i!=1  SS[nd,nd-ny*nz] = Sv.value[1] end
        if i!=nx SS[nd,nd+ny*nz] = Sv.value[7] end
        if j!=1  SS[nd,nd-nz] = Sv.value[2] end
        if j!=nx SS[nd,nd+nz] = Sv.value[6] end
        if k!=1  SS[nd,nd-1]  = Sv.value[3] end
        if k!=ny SS[nd,nd+1]  = Sv.value[5] end
    end
end=#
