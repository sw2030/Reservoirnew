import Base.zero, Base.+, Base.-, Base.*, Base./, Base.@propagate_inbounds

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

Stencil(x::Array{StencilPoint{T,N,P},N}) where {T,N,P} = Stencil{T,N,P,typeof(x)}(x)
MStencil(x::NTuple{MM,Stencil{T,N,P,A}}) where {MM,T,N,P,A} = MStencil{MM,T,N,P,A}(x)

(::Sindx{1,3})() = ((-1,), (0,), (1,))
(::Sindx{2,5})() = ((-1,0),(0,-1),(0,0),(0,1),(1,0))
(::Sindx{3,7})() = ((-1,0,0),(0,-1,0),(0,0,-1),(0,0,0),(0,0,1),(0,1,0),(1,0,0))


Base.iterate(S::StencilPoint{T,N,P}, state = 1) where {T,N,P} = state > P ? nothing : ((Sindx{N,P}()()[state], S.value[state]), state + 1)
#= 0.6 version
Base.start(S::StencilPoint)                                      = 1
@inbounds Base.next(S::StencilPoint{T,N,P}, state) where {T,N,P} = ((Sindx{N,P}()()[state], S.value[state]), state + 1)
Base.done(S::StencilPoint{T,N,P}, state) where {T,N,P}           = state > P
=#
Base.eltype(::Type{StencilPoint{T,N,P}}) where {T,N,P} = T

Base.copy(S::Stencil{T,N,P,A}) where {T,N,P,A}         = Stencil{T,N,P,A}(copy(S.v))
Base.copy(MS::MStencil{MM,T,N,P,A}) where {MM,T,N,P,A} = MStencil{MM,T,N,P,A}(copy.(MS.stencils))
Base.copy(g::Grid{T,N,P,S}) where {T,N,P,S}                                  = Grid{T,N,P,S}(copy(g.A))
Base.copyto!(g1::Grid{T,N,P,S}, g2::Grid{T,N,P,S}) where {T,N,P,S}           = copyto!(g1.A, g2.A)
Base.copy(Mg::MGrid{M,T,N,P,S}) where {M,T,N,P,S}                            = copy.(Mg)
Base.copyto!(Mg1::MGrid{M,T,N,P,S}, Mg2::MGrid{M,T,N,P,S}) where {M,T,N,P,S} = copyto!.(Mg1,Mg2)
LinearAlgebra.rmul!(g::Grid, a::Number)            = LinearAlgebra.rmul!(g.A, a)
LinearAlgebra.rmul!(Mg::MGrid, a::Number)          = LinearAlgebra.rmul!.(Mg, a)
LinearAlgebra.axpy!(a::Number,Mg1::MGrid{M,T,N,P,S}, Mg2::MGrid{M,T,N,P,S}) where {M,T,N,P,S} = LinearAlgebra.axpy!.(a,Mg1,Mg2)

@propagate_inbounds Base.getindex(g::Grid{T,1,3}, i) where {T}             = g.A[i+1]
@propagate_inbounds Base.getindex(g::Grid{T,2,5}, i, j) where {T}          = g.A[i+1,j+1]
@propagate_inbounds Base.getindex(g::Grid{T,2,1}, i, j) where {T}          = g.A[i,j]
@propagate_inbounds Base.getindex(g::Grid{T,3,7}, i, j, k) where {T}       = g.A[i+1,j+1,k+1]
@propagate_inbounds Base.getindex(g::Grid{T,3,1}, i, j, k) where {T}       = g.A[i,j,k]
@propagate_inbounds Base.getindex(Mg::MGrid{M,T,3},i,j,k) where {M,T}      = [(Mg[d])[i,j,k] for d in 1:M]
@propagate_inbounds Base.getindex(Mg::MGrid{M,T,3},i,j,k,d) where {M,T}    = (Mg[d])[i,j,k]
@propagate_inbounds Base.getindex(S::Stencil{T,1},i) where {T}             = S.v[i]
@propagate_inbounds Base.getindex(S::Stencil{T,2},i,j) where {T}           = S.v[i,j]
@propagate_inbounds Base.getindex(S::Stencil{T,3},i,j,k) where {T}         = S.v[i, j, k]
@propagate_inbounds Base.getindex(S::MStencil{MM,T,3}, i) where {MM,T}     = S.stencils[i]
#Base.getindex{T,P}(g::Grid{T,2,P}, i, j) = g.A[i + (P >> 2), j + (P >> 2)]
#Base.getindex{T,P}(g::Grid{T,3,P}, i, j) = g.A[i + (P >> 2), j + (P >> 2), k+(P>>2)]

@propagate_inbounds Base.setindex!(g::Grid{T,1,3}, a, i) where {T}    = setindex!(g.A, a, i+1)
@propagate_inbounds Base.setindex!(g::Grid{T,2,5}, a, i, j) where {T} = setindex!(g.A, a, i+1, j+1)
@propagate_inbounds Base.setindex!(g::Grid{T,3,7}, a, i, j, k) where {T} = setindex!(g.A, a, i+1, j+1, k+1)

LinearAlgebra.norm(g::Grid{T,2,5}) where {T} = LinearAlgebra.norm(g.A[2:end-1, 2:end-1])
LinearAlgebra.norm(g::Grid{T,3,7}) where {T} = LinearAlgebra.norm(g.A[2:end-1, 2:end-1, 2:end-1])
LinearAlgebra.norm(g::MGrid) = LinearAlgebra.norm(map(norm,g))
LinearAlgebra.dot(x::Grid{T,2,5}, y::Grid{T,2,5}) where {T} = LinearAlgebra.dot(x.A[2:end-1, 2:end-1], y.A[2:end-1, 2:end-1])
LinearAlgebra.dot(x::Grid{T,3,7}, y::Grid{T,3,7}) where {T} = LinearAlgebra.dot(x.A[2:end-1, 2:end-1, 2:end-1], y.A[2:end-1, 2:end-1, 2:end-1])
LinearAlgebra.dot(Mg1::MGrid{M,T,N,P,S}, Mg2::MGrid{M,T,N,P,S}) where {M,T,N,P,S} = sum(LinearAlgebra.dot.(Mg1,Mg2))

zero(x::Grid{T,N,P}) where {T,N,P}              = Grid{T,N,P,typeof(x.A)}(zero(x.A))
zero(x::MGrid{2,T,N,P,S}) where {T,N,P,S}       = MGrid{2,T,N,P,S}((zero(x[1]), zero(x[1])))
zero(S::StencilPoint{T,N,P}) where {T,N,P}      = StencilPoint{T,N,P}(zero.(S.value))
+(x::Grid{T,N,P}, y::Grid{T,N,P}) where {T,N,P} = Grid{T,N,P,typeof(x.A)}(x.A+y.A)
-(x::Grid{T,N,P}, y::Grid{T,N,P}) where {T,N,P} = Grid{T,N,P,typeof(x.A)}(x.A-y.A)
-(x::Grid{T,N,P}) where {T,N,P}                 = Grid{T,N,P,typeof(x.A)}(-x.A)
*(x::Grid{T,N,P}, y::Float64) where {T,N,P}     = Grid{T,N,P,typeof(x.A)}(x.A*y)
*(y::Float64, x::Grid{T,N,P}) where {T,N,P}     = Grid{T,N,P,typeof(x.A)}(x.A*y)
/(x::Grid{T,N,P}, y::Float64) where {T,N,P}     = Grid{T,N,P,typeof(x.A)}(x.A/y)
+(x::MGrid{M,T,N,P},y::MGrid{M,T,N,P}) where {M,T,N,P} = x.+y
-(x::MGrid{M,T,N,P},y::MGrid{M,T,N,P}) where {M,T,N,P} = x.-y


Base.size(g::Grid{T,1,3}) where {T} = (length(g.A)-2, )
Base.size(g::Grid{T,2,1}) where {T} = size(g.A)
Base.size(g::Grid{T,2,5}) where {T} = (size(g.A, 1)-2, size(g.A, 2)-2)
Base.size(g::Grid{T,3,1}) where {T} = size(g.A)
Base.size(g::Grid{T,3,7}) where {T} = (size(g.A, 1)-2, size(g.A, 2)-2, size(g.A, 3)-2)
Base.size(S::Stencil)               = size(S.v)
gridsize(g::Grid)                   = prod(size(g))
gridsize(g::MGrid{M}) where {M}     = prod(size(g[1]))*M


function makegrid(x::Array{T,1},P) where {T}
    r = Grid{T,1,P}(zeros(eltype(x), length(x)+2))
    r.A[2:end-1] = x
    r
end
function makegrid(x::Array{T,2},P) where {T}
    r = Grid{T,2,P,Array{T,2}}(zeros(eltype(x),size(x,1)+2,size(x,2)+2))
    r.A[2:end-1, 2:end-1] = x
    r
end
function makegrid(x::Array{T,3},P) where {T}
    r = Grid{T,3,P,Array{T,3}}(zeros(eltype(x),size(x,1)+2,size(x,2)+2,size(x,3)+2))
    r.A[2:end-1, 2:end-1, 2:end-1] = x
    r
end


function A_mul_B!(a::Number, S::Stencil{TS,1}, x::Grid{Txy,1}, b::Number, y::Grid{Txy,1}) where {Txy,TS}
    if b!= 1
        b != 0 ? LinearAlgebra.rmul!(y.A, b) : fill!(y.A, zero(Txy))
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
function A_mul_B!(a::Number, S::Stencil{TS,2}, x::Grid{Txy,2},  b::Number, y::Grid{Txy,2}) where {Txy,TS}
    if b!= 1
        b != 0 ? LinearAlgebra.rmul!(y.A, b) : fill!(y.A, zero(Txy))
    end
    for j = 1:size(S.v)[2], i = 1:size(S.v)[1]
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
function A_mul_B!(a::Number, S::Stencil{TS,3,P}, x::Grid{Txy,3,P},  b::Number, y::Grid{Txy,3,P}) where {Txy,TS,P}
    if b!= 1
        b != 0 ? LinearAlgebra.rmul!(y.A, b) : fill!(y.A, zero(Txy))
    end
    Sid = Sindx{3,P}()()
    @inbounds for k in axes(S.v,3), j in axes(S.v,2), i in axes(S.v,1)
        tmp = zero(Txy)
        Sijk = S[i,j,k].value # with inbounds -0.3ms
        for c in 1:P
            tmp += Sijk[c]*x[i+Sid[c][1], j+Sid[c][2], k+Sid[c][3]]
        end
        y[i,j,k] += a*tmp
    end
    return y
end
## A*b version
function A_mul_B!(S::Stencil{TS,3,P}, x::Grid{Txy,3,P}, y::Grid{Txy,3,P}) where {Txy,TS,P}
    Sid = Sindx{3,P}()()
    @inbounds for k in axes(S.v,3), j in axes(S.v,2), i in axes(S.v,1)
        tmp = zero(Txy)
        Sijk = S[i,j,k].value # with inbounds -0.3ms
        for c in 1:P
            tmp += Sijk[c]*x[i+Sid[c][1], j+Sid[c][2], k+Sid[c][3]]
        end
        y[i,j,k] += tmp
    end
    return y
end

## Full A_mul_B case for MStencil
function A_mul_B!(a::Number, MS::MStencil{4,TS,3}, x::MGrid{2,Txy,3},  b::Number, y::MGrid{2,Txy,3}) where {Txy,TS}
    A_mul_B!(a,MS[1],x[1],b,y[1])
    A_mul_B!(a,MS[2],x[2],1,y[1])
    A_mul_B!(a,MS[3],x[1],b,y[2])
    A_mul_B!(a,MS[4],x[2],1,y[2])
    return y
end
# A*b version
function A_mul_B!(MS::MStencil{4,TS,3}, x::MGrid{2,Txy,3}, y::MGrid{2,Txy,3}) where {TS,Txy}
    A_mul_B!(MS[1],x[1],y[1])
    A_mul_B!(MS[2],x[2],y[1])
    A_mul_B!(MS[3],x[1],y[2])
    A_mul_B!(MS[4],x[2],y[2])
    return y
end

Base.:*(S::Stencil{TS,N,P}, x::Grid{Tx,N,P,A}) where {TS,Tx,N,P,A}              = A_mul_B!(S, x,zero(x))
Base.:*(MS::MStencil{MM,TS,N,P}, x::MGrid{M,Tx,N,P,A}) where {MM,M,TS,Tx,N,P,A} = A_mul_B!(MS,x,zero(x))

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
