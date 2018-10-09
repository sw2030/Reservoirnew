const DGrid{T,N,P,S} = Grid{T,N,1,DArray{T,N,Grid{T,N,P,S}}}
#const DMGrid{M,T,N,P,S} = NTuple{M,Grid{T,N,1,DArray{T,N,Grid{T,N,P,S}}}}
const DMGrid{M,T,N,P,S} = MGrid{M,T,N,1,DArray{T,N,Grid{T,N,P,S}}}
const DStencil{T,N,P} = Stencil{T,N,P,<:DArray}
const DMStencil{MM,T,N,P} = MStencil{MM,T,N,P,<:DArray}

function A_mul_B!(a::Number, S::DStencil{TS,3}, x::DGrid{Txy,3},  b::Number, y::DGrid{Txy,3}) where {TS,Txy}
    @sync begin
        for id in procs(y.A)
            @async remotecall_fetch(id) do
                A_mul_B!(a, localpart(S), localpart(x), b, localpart(y))
                nothing
            end
        end
    end
    @sync begin
        for id in procs(y.A)
            @async remotecall_fetch(gridsync, id, id, y.A)
        end
    end
    return y
end
### Full A_mul_B! case
function A_mul_B!(a::Number, MS::DMStencil{4,TS,3}, x::DMGrid{2,Txy,3},  b::Number, y::DMGrid{2,Txy,3}) where {Txy,TS}
    A_mul_B!(a,MS[1],x[1],b,y[1])
    A_mul_B!(a,MS[2],x[2],1,y[1])
    A_mul_B!(a,MS[3],x[1],b,y[2])
    A_mul_B!(a,MS[4],x[2],1,y[2])
    return y
end
### A*b version
function A_mul_B!(S::DMStencil{4,TS,3}, x::DMGrid{2,Txy,3}, y::DMGrid{2,Txy,3}) where {TS,Txy}
    @sync begin
        for id in procs(y[1].A)
            @async remotecall_fetch(id) do
                A_mul_B!(localpart(S), localpart(x), localpart(y))
                nothing
            end
        end
    end
    @sync begin
        for id in procs(y[1].A)
            @async remotecall_fetch(id) do
                gridsync(id, y[1].A)
                gridsync(id, y[2].A)
            end
        end
    end
    y
end
function Base.:*(S::DStencil{TS,N,P},x::DGrid{Tx,N,P,SS}) where {TS,N,Tx,P,SS}
    D = DistributedArrays.DArray(I -> (*(localpart(S), localpart(x))), S.v)
    @sync begin
        for id in procs(D)
            @async remotecall_fetch(gridsync, id, id, D)
        end
    end
    return Grid{Tx,N,1,DistributedArrays.DArray{Tx,N,Grid{Tx,N,P,SS}}}(D)
end
Base.:*(S::DMStencil{MM,TS,N,P},x::DMGrid{M,Txy,N,P,SS}) where {MM,M,TS,Txy,N,P,SS} = A_mul_B!(S, x, zero(x))

function LinearAlgebra.dot(D1::DistributedArrays.DArray{T}, D2::DistributedArrays.DArray{T})::float(eltype(T)) where {T}
    r = asyncmap(procs(D1)) do p
        remotecall_fetch(p) do
            LinearAlgebra.dot(localpart(D1), localpart(D2))
        end
    end
    return sum(r)
end

LinearAlgebra.norm(g::DGrid) = LinearAlgebra.norm(g.A)
LinearAlgebra.norm(g::DMGrid) = LinearAlgebra.norm(LinearAlgebra.norm.(g))
LinearAlgebra.dot(gx::DGrid{T,N,P,S}, gy::DGrid{T,N,P,S}) where {T,N,P,S}            = LinearAlgebra.dot(gx.A, gy.A)
LinearAlgebra.dot(gx::DMGrid{M,T,N,P,S}, gy::DMGrid{M,T,N,P,S}) where {M,T,N,P,S}    = sum(LinearAlgebra.dot.(gx, gy))

Base.copy(g::DGrid{T,N,P,S}) where {T,N,P,S} = DGrid{T,N,P,S}(DArray(I->copy(localpart(g)), g.A))
function Base.copyto!(g1::DGrid{T,N,P,S}, g2::DGrid{T,N,P,S}) where {T,N,P,S}
    @sync begin
        for id in procs(g1.A)
            @async remotecall_fetch(id) do
                copyto!(localpart(g1), localpart(g2))
                nothing
            end
        end
    end
    g2
end
function LinearAlgebra.rmul!(g::DGrid, a::Number)
    @sync begin
        for id in procs(g.A)
            @async remotecall_fetch(id) do
                LinearAlgebra.rmul!(localpart(g).A, a)
                nothing
            end
        end
    end
    return g
end

zero(g::DGrid{T,N,P,S}) where {T,N,P,S} = Grid{T,N,1,DArray{T,N,Grid{T,N,P,S}}}(DArray(I->zero(localpart(g)), g.A))
zero(mg::DMGrid{M,T,N,P,S}) where {M,T,N,P,S}  = zero.(mg)
function /(g::DGrid{T,N,P,S}, x::Float64) where {T,N,P,S}
    D = DistributedArrays.DArray(I-> /(localpart(g.A),x), g.A)
    Grid{T,N,1,typeof(D)}(D)
end
function *(x::Float64, g::DGrid{T,N,P,S}) where {T,N,P,S}
    D = DistributedArrays.DArray(I-> *(localpart(g.A),x), g.A)
    Grid{T,N,1,typeof(D)}(D)
end
function *(g::DGrid{T,N,P,S}, x::Float64) where {T,N,P,S}
    D = DistributedArrays.DArray(I-> *(x,localpart(g.A)), g.A)
    Grid{T,N,1,typeof(D)}(D)
end
function LinearAlgebra.axpy!(a::Number, g1::DGrid{T,N,P,S}, g2::DGrid{T,N,P,S}) where {T,N,P,S}
    @sync begin
        for id in procs(g1.A)
            @async remotecall_fetch(id) do
                LinearAlgebra.axpy!(a, localpart(g1.A), localpart(g2.A))
                nothing
            end
        end
    end
    g2
end


function gridsync(id, D::DistributedArrays.DArray{T,2}) where {T}
    s = size(D.pids)
    i,j = Tuple(CartesianIndices(s)[findfirst(isequal(id),D.pids)])#ind2sub(s, findfirst(isequal(id),D.pids))
    if i!=1    remotecall_fetch(t->localpart(t).A[1,:] = remotecall_fetch(m->localpart(m).A[end-1,:], t.pids[LinearIndices(s)[i-1,j]],t),id,D) end
    if i!=s[1] remotecall_fetch(t->localpart(t).A[end,:] = remotecall_fetch(m->localpart(m).A[2,:], t.pids[LinearIndices(s)[i+1,j]],t),id,D) end
    if j!=1    remotecall_fetch(t->localpart(t).A[:,1] = remotecall_fetch(m->localpart(m).A[:,end-1], t.pids[LinearIndices(s)[i,j-1]],t),id,D) end
    if j!=s[2] remotecall_fetch(t->localpart(t).A[:,end] = remotecall_fetch(m->localpart(m).A[:,2], t.pids[LinearIndices(s)[i,j+1]],t),id,D) end
end
function gridsync(id, D::DistributedArrays.DArray{T,3}) where {T}
    s = size(D.pids)
    i,j,k = Tuple(CartesianIndices(s)[findfirst(isequal(id),D.pids)])
    if i!=1    remotecall_fetch(t->localpart(t).A[1,:,:] = remotecall_fetch(m->localpart(m).A[end-1,:,:], t.pids[LinearIndices(s)[i-1,j,k]],t),id,D) end
    if i!=s[1] remotecall_fetch(t->localpart(t).A[end,:,:] = remotecall_fetch(m->localpart(m).A[2,:,:], t.pids[LinearIndices(s)[i+1,j,k]],t),id,D) end
    if j!=1    remotecall_fetch(t->localpart(t).A[:,1,:] = remotecall_fetch(m->localpart(m).A[:,end-1,:], t.pids[LinearIndices(s)[i,j-1,k]],t),id,D) end
    if j!=s[2] remotecall_fetch(t->localpart(t).A[:,end,:] = remotecall_fetch(m->localpart(m).A[:,2,:], t.pids[LinearIndices(s)[i,j+1,k]],t),id,D) end
    if k!=1    remotecall_fetch(t->localpart(t).A[:,:,1] = remotecall_fetch(m->localpart(m).A[:,:,end-1], t.pids[LinearIndices(s)[i,j,k-1]],t),id,D) end
    if k!=s[3] remotecall_fetch(t->localpart(t).A[:,:,end] = remotecall_fetch(m->localpart(m).A[:,:,2], t.pids[LinearIndices(s)[i,j,k+1]],t),id,D) end
end


function makegrid(x::DistributedArrays.DArray{T,N},P) where {T,N}## Take DArray
    D = DistributedArrays.DArray(I -> makegrid(DistributedArrays.localpart(x),P), x) ## Make DArray of Grids
    @sync begin
        for id in procs(D)
            @async remotecall_fetch(gridsync, id, id, D)
        end
    end
    return Grid{T,N,1,typeof(D)}(D)
end
#makegrid{M,T,N}(x::NTuple{M,DistributedArrays.DArray{T,N}},P) = makegrid.(x,P)

#### IS this okay? type recognition inside constructor might be bad idea.
DistributedArrays.localpart(G::DGrid) = localpart(G.A)
DistributedArrays.localpart(G::DMGrid) = map(localpart, G) ## localpart(DMG) = (localpart(DG1), localpart(DG2),...)
DistributedArrays.localpart(S::DStencil) = Stencil(localpart(S.v))
DistributedArrays.localpart(S::DMStencil) = MStencil(map(localpart,S.stencils))
#= function DistributedArrays.localpart{TS,N,P}(S::DStencil{TS,N,P})
    localS = localpart(S.v)
    Stencil{TS,N,P,typeof(localS)}(localS)
end
function DistributedArrays.localpart{MM,TS,N,P}(MS::DMStencil{MM,TS,N,P})
    localMS = localpart.(MS.stencils) ## returns (::Stencil, ...)
    MStencil{MM,TS,N,P,}(localpart.(MS))
end =#
function DistributedArrays.localpart(m::Reservoirmodel{S}) where {S<:DistributedArrays.DArray}
    Reservoirmodel(
        m.Δt,
        m.Tf,
        localpart.(m.Δ),
        localpart(m.z),
        localpart(m.k),
        localpart(m.p_ref),
        m.C_r,
        m.ϕ_ref,
        localpart(m.ϕ),
        m.k_r_w,
        m.k_r_o,
        m.p_cow,
        m.C_w,
        m.C_o,
        m.ρ_w,
        m.ρ_o,
        m.μ_w,
        m.μ_o)
end
function getresidual(m, q, g::DMGrid{M,T,N,P,S}, g_prev::DMGrid{M,T,N,P,S}) where {M,T,N,P,S}
    D = DistributedArrays.DArray(I-> getlocalresidual(localpart(m), localpart(q), localpart(g), localpart(g_prev)), m.z)
    D1 = DArray(I->makegrid(map(t->t[1], localpart(D)),7), D)
    D2 = DArray(I->makegrid(map(t->t[2], localpart(D)),7), D)
    @sync begin
        for id in procs(D)
            @async remotecall_fetch(id) do
                gridsync(id, D1)
                gridsync(id, D2)
            end
        end
    end
    return (Grid{T,N,1,typeof(D1)}(D1),Grid{T,N,1,typeof(D2)}(D2))
end
function getlocalstencil(m, q, g::MGrid{M,T,N,P,S}, g_prev::MGrid{M,T,N,P,S}) where {M,T,N,P,S}
    Nx, Ny, Nz = size(m)
    stencilArray = Array{NTuple{4,StencilPoint{Float64,3,7}},3}(undef,Nx, Ny, Nz)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        J = ForwardDiff.jacobian(θ -> res_f(m, q[i,j,k,:], θ, g_prev, i, j, k), [g[i-1,j,k,1] g[i-1,j,k,2]; g[i,j-1,k,1] g[i,j-1,k,2];
                                                                                    g[i,j,k-1,1] g[i,j,k-1,2]; g[i,j,k,1] g[i,j,k,2];
                                                                                    g[i,j,k+1,1] g[i,j,k+1,2]; g[i,j+1,k,1] g[i,j+1,k,2]; g[i+1,j,k,1] g[i+1,j,k,2]])
        stencilArray[i,j,k] = (StencilPoint{Float64,3,7}((J[1,1],J[1,2],J[1,3],J[1,4],J[1,5],J[1,6],J[1,7])), StencilPoint{Float64,3,7}((J[1,8],J[1,9],J[1,10],J[1,11],J[1,12],J[1,13],J[1,14])),
                    StencilPoint{Float64,3,7}((J[2,1],J[2,2],J[2,3],J[2,4],J[2,5],J[2,6],J[2,7])), StencilPoint{Float64,3,7}((J[2,8],J[2,9],J[2,10],J[2,11],J[2,12],J[2,13],J[2,14])))
    end
    return stencilArray
end
function getstencil(m, q, g::DMGrid{M,T,N,P,S}, g_prev::DMGrid{M,T,N,P,S}) where {M,T,N,P,S}
    D = DArray(I->getlocalstencil(localpart(m), localpart(q), localpart(g), localpart(g_prev)), m.z)
    D1 = DArray(I->map(t->t[1],localpart(D)), D)
    D2 = DArray(I->map(t->t[2],localpart(D)), D)
    D3 = DArray(I->map(t->t[3],localpart(D)), D)
    D4 = DArray(I->map(t->t[4],localpart(D)), D)
    return MStencil((Stencil{T,N,P,typeof(D1)}(D1), Stencil{T,N,P,typeof(D1)}(D2), Stencil{T,N,P,typeof(D1)}(D3), Stencil{T,N,P,typeof(D1)}(D4)))
end
function make_P_E_precond_1(DMS::DMStencil{4,Float64,3,7})
    rrs = [@spawnat p make_P_E_precond_1(localpart(DMS))[3:4] for p in workers()]
    PDarrays = [DArray(DMS.stencils[1].v) do I
                    fetch(rrs[myid()-1])[1][i]
                end for i in 1:4]
    EDarrays = [DArray(DMS.stencils[1].v) do I
                    fetch(rrs[myid()-1])[2][i]
                end for i in 1:4]
    return MStencil{4,Float64,3,7,typeof(PDarrays[1])}(Tuple(Stencil{Float64,3,7,typeof(PDarrays[1])}.(PDarrays))),
                MStencil{4,Float64,3,7,typeof(EDarrays[1])}(Tuple(Stencil{Float64,3,7,typeof(EDarrays[1])}.(EDarrays)))
end
