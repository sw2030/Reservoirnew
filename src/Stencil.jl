import Base.zero
import Base.+
import Base.-
import Base.*
import Base./

struct Sindx{N,P} end
struct StencilPoint{T,N,P}
        value::NTuple{P,T}
end
struct Stencil{T,N,P,A}
    v::A #Array{StencilPoint{T,N,P},N}
end
struct MStencil{MM,T,N,P,A}
    stencils::NTuple{MM, Stencil{T,N,P,A}}
end
struct Grid{T,N,P,S<:AbstractArray}<:AbstractArray{T,N}
    A::S
end
const MGrid{M,T,N,P,S}        = NTuple{M, Grid{T,N,P,S}}


(::Sindx{1,3})() = ((-1,), (0,), (1,))
(::Sindx{2,5})() = ((-1,0),(0,-1),(0,0),(0,1),(1,0))
(::Sindx{3,7})() = ((-1,0,0),(0,-1,0),(0,0,-1),(0,0,0),(0,0,1),(0,1,0),(1,0,0))


Base.start(S::StencilPoint)                               = 1
@inbounds Base.next{T,N,P}(S::StencilPoint{T,N,P}, state) = ((Sindx{N,P}()()[state], S.value[state]), state + 1)
Base.done{T,N,P}(S::StencilPoint{T,N,P}, state)           = state > P


Base.copy{T,N,P,A}(S::Stencil{T,N,P,A})         = Stencil{T,N,P,A}(copy(S.v))
Base.copy{MM,T,N,P,A}(MS::MStencil{MM,T,N,P,A}) = MStencil{MM,T,N,P,A}(copy.(MS.stencils))
Base.getindex{T}(S::Stencil{T,1},i)        = S.v[i]
Base.getindex{T}(S::Stencil{T,2},i,j)      = S.v[i,j]
Base.getindex{T}(S::Stencil{T,3},i,j,k)    = S.v[i, j, k]
Base.getindex{MM,T}(S::MStencil{MM,T,3}, i) = S.stencils[i]
Base.eltype{T,N,P}(::Type{StencilPoint{T,N,P}}) = T

Base.copy{T,N,P,S}(g::Grid{T,N,P,S})                                  = Grid{T,N,P,S}(copy(g.A))
Base.copy!{T,N,P,S}(g1::Grid{T,N,P,S}, g2::Grid{T,N,P,S})             = copy!(g1.A, g2.A)
Base.copy{M,T,N,P,S}(Mg::MGrid{M,T,N,P,S})                            = copy.(Mg)
Base.copy!{M,T,N,P,S}(Mg1::MGrid{M,T,N,P,S}, Mg2::MGrid{M,T,N,P,S})   = copy!.(Mg1,Mg2)
Base.scale!{M,T,N,P,S}(Mg::MGrid{M,T,N,P,S}, a::Number)               = scale!.(Mg, a)
Base.dot{M,T,N,P,S}(Mg1::MGrid{M,T,N,P,S}, Mg2::MGrid{M,T,N,P,S})     = sum(dot.(Mg1,Mg2))
LinAlg.axpy!{M,T,N,P,S}(a::Number,Mg1::MGrid{M,T,N,P,S}, Mg2::MGrid{M,T,N,P,S}) = LinAlg.axpy!.(a,Mg1,Mg2)

Base.getindex{T}(g::Grid{T,1,3}, i)          = g.A[i+1]
Base.getindex{T}(g::Grid{T,2,5}, i, j)       = g.A[i+1,j+1]
Base.getindex{T}(g::Grid{T,2,1}, i, j)       = g.A[i,j]
Base.getindex{T}(g::Grid{T,3,7}, i, j, k)    = g.A[i+1,j+1,k+1]
Base.getindex{T}(g::Grid{T,3,1}, i, j, k)    = g.A[i,j,k]
Base.getindex{M,T}(Mg::MGrid{M,T,3},i,j,k)   = [(Mg[d])[i,j,k] for d in 1:M]
Base.getindex{M,T}(Mg::MGrid{M,T,3},i,j,k,d) = (Mg[d])[i,j,k]
#Base.getindex{T,P}(g::Grid{T,2,P}, i, j) = g.A[i + (P >> 2), j + (P >> 2)]
#Base.getindex{T,P}(g::Grid{T,3,P}, i, j) = g.A[i + (P >> 2), j + (P >> 2), k+(P>>2)]


function Base.setindex!{T}(g::Grid{T,1,3}, a, i)
    setindex!(g.A, a, i+1)
end
function Base.setindex!{T}(g::Grid{T,2,5}, a, i, j)
    setindex!(g.A, a, i+1, j+1)
end
function Base.setindex!{T}(g::Grid{T,3,7}, a, i, j, k)
    setindex!(g.A, a, i+1, j+1, k+1)
end

Base.norm{T}(g::Grid{T,2,5}) = Base.LinAlg.vecnorm2(g.A[2:end-1, 2:end-1])
Base.norm{T}(g::Grid{T,3,7}) = Base.LinAlg.vecnorm2(g.A[2:end-1, 2:end-1, 2:end-1])
Base.norm(g::MGrid) = Base.LinAlg.vecnorm2(Base.LinAlg.vecnorm2.(g))
Base.dot{T}(x::Grid{T,2,5}, y::Grid{T,2,5}) = vecdot(x.A[2:end-1, 2:end-1], y.A[2:end-1, 2:end-1])
Base.dot{T}(x::Grid{T,3,7}, y::Grid{T,3,7}) = vecdot(x.A[2:end-1, 2:end-1, 2:end-1], y.A[2:end-1, 2:end-1, 2:end-1])

zero{T,N,P}(x::Grid{T,N,P})              = Grid{T,N,P,typeof(x.A)}(zero(x.A))
zero{T,N,P,S}(x::MGrid{2,T,N,P,S})       = MGrid{2,T,N,P,S}((zero(x[1]), zero(x[1])))
zero{T,N,P}(S::StencilPoint{T,N,P})      = StencilPoint{T,N,P}(zero.(S.value))
+{T,N,P}(x::Grid{T,N,P}, y::Grid{T,N,P}) = Grid{T,N,P,typeof(x.A)}(x.A+y.A)
-{T,N,P}(x::Grid{T,N,P}, y::Grid{T,N,P}) = Grid{T,N,P,typeof(x.A)}(x.A-y.A)
-{T,N,P}(x::Grid{T,N,P})                 = Grid{T,N,P,typeof(x.A)}(-x.A)
*{T,N,P}(x::Grid{T,N,P}, y::Float64)     = Grid{T,N,P,typeof(x.A)}(x.A*y)
*{T,N,P}(y::Float64, x::Grid{T,N,P})     = Grid{T,N,P,typeof(x.A)}(x.A*y)
/{T,N,P}(x::Grid{T,N,P}, y::Float64)     = Grid{T,N,P,typeof(x.A)}(x.A/y)
-{M,T,N,P}(x::MGrid{M,T,N,P},y::MGrid{M,T,N,P}) = x.-y

Base.size{T}(g::Grid{T,1,3}) = (length(g.A)-2, )
Base.size{T}(g::Grid{T,2,1}) = size(g.A)
Base.size{T}(g::Grid{T,2,5}) = (size(g.A, 1)-2, size(g.A, 2)-2)
Base.size{T}(g::Grid{T,3,1}) = size(g.A)
Base.size{T}(g::Grid{T,3,7}) = (size(g.A, 1)-2, size(g.A, 2)-2, size(g.A, 3)-2)
Base.size{T,N,P}(S::Stencil{T,N,P}) = size(S.v)
Base.size{MM,T,N,P}(MS::MStencil{MM,T,N,P}) = size(MS.stencils[1])
gridsize(g::Grid) = prod(size(g))
gridsize{M}(g::MGrid{M}) = prod(size(g[1]))*M


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


function Base.A_mul_B!{Txy,TS}(a::Number, S::Stencil{TS,1}, x::Grid{Txy,1}, b::Number, y::Grid{Txy,1})
    if b!= 1
        b != 0 ? scale!(y.A, b) : fill!(y.A, zero(Txy))
    end
    for i in 1:size(S.v)[1]
        tmp = zero(Txy)
        @inbounds for (idx, value) in S[i]
            (ix, ) = idx
            tmp    += value*x[i+ix]
        end
        y[i] += a*tmp
    end
    return y
end
function Base.A_mul_B!{Txy,TS}(a::Number, S::Stencil{TS,2}, x::Grid{Txy,2},  b::Number, y::Grid{Txy,2})
    if b!= 1
        b != 0 ? scale!(y.A, b) : fill!(y.A, zero(Txy))
    end
    for j = 1:size(S.v)[1], j = 1:size(S.v)[2]
        tmp = zero(Txy)
        @inbounds for (idx, value) in S[i,j]
            ix, jy = idx
            tmp   +=  value*x[i+ix, j+jy]
        end
        y[i,j] += a*tmp
    end
    return y
end

## Full A_mul_B case
function Base.A_mul_B!(a::Number, S::Stencil{TS,3}, x::Grid{Txy,3},  b::Number, y::Grid{Txy,3}) where {Txy,TS}
    if b!= 1
        b != 0 ? scale!(y.A, b) : fill!(y.A, zero(Txy))
    end
    for k in 1:size(S.v)[1], j in 1:size(S.v)[2], i in 1:size(S.v)[3]
        tmp = zero(Txy)
        @inbounds for (idx, value) in S[i,j,k]
            ix, jy, kz = idx
            tmp += value*x[i+ix, j+jy, k+kz]
        end
        y[i,j,k] += a*tmp
    end
    return y
end
## A*b version
function Base.A_mul_B!(S::Stencil{TS,3}, x::Grid{Txy,3}, y::Grid{Txy,3}) where {Txy,TS}
    for k in 1:size(S.v)[1], j in 1:size(S.v)[2], i in 1:size(S.v)[3]
        tmp = zero(Txy)
        @inbounds for (idx, value) in S[i,j,k]
            ix, jy, kz = idx
            tmp += value*x[i+ix, j+jy, k+kz]
        end
        y[i,j,k] += tmp
    end
    return y
end

## Full A_mul_B case for MStencil
function Base.A_mul_B!(a::Number, MS::MStencil{4,TS,3}, x::MGrid{2,Txy,3},  b::Number, y::MGrid{2,Txy,3}) where {Txy,TS}
    A_mul_B!(a,MS[1],x[1],b,y[1])
    A_mul_B!(a,MS[2],x[2],1,y[1])
    A_mul_B!(a,MS[3],x[1],b,y[2])
    A_mul_B!(a,MS[4],x[2],1,y[2])
    return y
end
# A*b version
function Base.A_mul_B!(MS::MStencil{4,TS,3}, x::MGrid{2,Txy,3}, y::MGrid{2,Txy,3}) where {TS,Txy}
    A_mul_B!(MS[1],x[1],y[1])
    A_mul_B!(MS[2],x[2],y[1])
    A_mul_B!(MS[3],x[1],y[2])
    A_mul_B!(MS[4],x[2],y[2])
    return y
end

Base.:*{TS,Tx,N,P,A}(S::Stencil{TS,N,P}, x::Grid{Tx,N,P,A})              = A_mul_B!(S, x,zero(x))
Base.:*{MM,M,TS,Tx,N,P,A}(MS::MStencil{MM,TS,N,P}, x::MGrid{M,Tx,N,P,A}) = A_mul_B!(MS,x,zero(x))


function Base.LinAlg.axpy!{T,N,P,S}(a, D1::Grid{T,N,P,S}, D2::Grid{T,N,P,S})
    LinAlg.axpy!(a, D1.A, D2.A)
    return D2
end
function Base.LinAlg.scale!(D::Grid, a::Number)
    LinAlg.scale!(D.A, a)
    return D
end






### EXTRA SAVE
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
