const DGrid{T,N,P,S} = Grid{T,N,1,DistributedArrays.DArray{T,N,Grid{T,N,P,S}}}
const DStencil{T,N,P} = Stencil{T,N,P,<:DistributedArrays.DArray}

function Base.A_mul_B!{TS,N,Txy,P}(a::Number, S::DStencil{TS,N,P}, x::DGrid{Txy,N},  b::Number, y::DGrid{Txy,N})
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
function Base.:*{TS,N,Tx,P,SS}(S::DStencil{TS,N,P},x::DGrid{Tx,N,P,SS})
    D = DistributedArrays.DArray(I -> (*(localpart(S), localpart(x))), S.v)
    @sync begin
        for id in procs(D)
            @async remotecall_fetch(gridsync, id, id, D)
        end
    end
    return Grid{Tx,N,1,DistributedArrays.DArray{Tx,N,Grid{Tx,N,P,SS}}}(D)
end
function Base.LinAlg.vecnorm{T}(D::DistributedArrays.DArray{T})::real(float(eltype(T)))
    r = asyncmap(procs(D)) do p
        remotecall_fetch(p) do
            vecnorm(localpart(D))
        end
    end
    return vecnorm(r)
end
function Base.LinAlg.vecdot{T}(D1::DistributedArrays.DArray{T}, D2::DistributedArrays.DArray{T})::float(eltype(T))
    r = asyncmap(procs(D1)) do p
        remotecall_fetch(p) do
            vecdot(localpart(D1), localpart(D2))
        end
    end
    return sum(r)
end
Base.norm(g::DGrid) = vecnorm(g.A)
Base.dot{T,N,P,S}(gx::DGrid{T,N,P,S}, gy::DGrid{T,N,P,S}) = vecdot(gx.A, gy.A)

function Base.scale!(g::DGrid, a::Number)
    @sync begin
        for id in procs(g.A)
            @async remotecall_fetch(id) do
                scale!(localpart(g).A, a)
                nothing
            end
        end
    end
    return g
end
function zero{T,N,P,S}(g::DGrid{T,N,P,S})
    D = DArray(I->zero(localpart(g)), g.A)
    return Grid{T,N,1,DistributedArrays.DArray{T,N,Grid{T,N,P,S}}}(D)
end
function /{T,N,P,S}(g::DGrid{T,N,P,S}, x::Float64)
    D = DistributedArrays.DArray(I-> /(localpart(g.A),x), g.A)
    Grid{T,N,1,typeof(D)}(D)
end
function *{T,N,P,S}(x::Float64, g::DGrid{T,N,P,S})
    D = DistributedArrays.DArray(I-> *(localpart(g.A),x), g.A)
    Grid{T,N,1,typeof(D)}(D)
end
function *{T,N,P,S}(g::DGrid{T,N,P,S}, x::Float64)
    D = DistributedArrays.DArray(I-> *(x,localpart(g.A)), g.A)
    Grid{T,N,1,typeof(D)}(D)
end

function gridsync{T}(id, D::DistributedArrays.DArray{T,2})
    s = size(D.pids)
    i,j = ind2sub(s, findfirst(D.pids,id))
    if i!=1    remotecall_fetch(t->localpart(t).A[1,:] = remotecall_fetch(m->localpart(m).A[end-1,:], t.pids[sub2ind(s,i-1,j)],t),id,D) end
    if i!=s[1] remotecall_fetch(t->localpart(t).A[end,:] = remotecall_fetch(m->localpart(m).A[2,:], t.pids[sub2ind(s,i+1,j)],t),id,D) end
    if j!=1    remotecall_fetch(t->localpart(t).A[:,1] = remotecall_fetch(m->localpart(m).A[:,end-1], t.pids[sub2ind(s,i,j-1)],t),id,D) end
    if j!=s[2] remotecall_fetch(t->localpart(t).A[:,end] = remotecall_fetch(m->localpart(m).A[:,2], t.pids[sub2ind(s,i,j+1)],t),id,D) end
end
function gridsync{T}(id, D::DistributedArrays.DArray{T,3})
    s = size(D.pids)
    i,j,k = ind2sub(s, findfirst(D.pids,id))
    if i!=1    remotecall_fetch(t->localpart(t).A[1,:,:] = remotecall_fetch(m->localpart(m).A[end-1,:,:], t.pids[sub2ind(s,i-1,j,k)],t),id,D) end
    if i!=s[1] remotecall_fetch(t->localpart(t).A[end,:,:] = remotecall_fetch(m->localpart(m).A[2,:,:], t.pids[sub2ind(s,i+1,j,k)],t),id,D) end
    if j!=1    remotecall_fetch(t->localpart(t).A[:,1,:] = remotecall_fetch(m->localpart(m).A[:,end-1,:], t.pids[sub2ind(s,i,j-1,k)],t),id,D) end
    if j!=s[2] remotecall_fetch(t->localpart(t).A[:,end,:] = remotecall_fetch(m->localpart(m).A[:,2,:], t.pids[sub2ind(s,i,j+1,k)],t),id,D) end
    if k!=1    remotecall_fetch(t->localpart(t).A[:,:,1] = remotecall_fetch(m->localpart(m).A[:,:,end-1], t.pids[sub2ind(s,i,j,k-1)],t),id,D) end
    if k!=s[3] remotecall_fetch(t->localpart(t).A[:,:,end] = remotecall_fetch(m->localpart(m).A[:,:,2], t.pids[sub2ind(s,i,j,k+1)],t),id,D) end
end
#=function syncgrod(id, D)
    s = size(D.pids)
    i,j = ind2sub(s, findfirst(D.pids,id))
    if j!=1
        remotecall_fetch(t->localpart(t).A[:,1] = remotecall_fetch(m->localpart(m).A[:,end-1], t.pids[sub2ind(s,i,j-1)], t),id,D)
    end
    if j!=s[2]
        remotecall_fetch(t->localpart(t).A[:,end] = remotecall_fetch(m->localpart(m).A[:,2], t.pids[sub2ind(s,i,j+1)], t),id,D)
    end
end
function rowsync(id, D)
    s = size(D.pids)
    i,j = ind2sub(s, findfirst(D.pids,id))
    if i!=1
        remotecall_fetch(t->localpart(t).A[1,:] = remotecall_fetch(m->localpart(m).A[end-1,:], t.pids[sub2ind(s,i-1,j)], t),id,D)
    end
    if i!=s[1]
        remotecall_fetch(t->localpart(t).A[end,:] = remotecall_fetch(m->localpart(m).A[2,:], t.pids[sub2ind(s,i+1,j)], t),id,D)
    end
end=#

function makegrid{T,N}(x::DistributedArrays.DArray{T,N},P)
    D = DistributedArrays.DArray(I -> makegrid(DistributedArrays.localpart(x),P), x)
    @sync begin
        for id in procs(D)
            @async remotecall_fetch(gridsync, id, id, D)
        end
    end
    return Grid{T,N,1,typeof(D)}(D)
end

function DistributedArrays.localpart(G::DGrid)
    return localpart(G.A)
end
function DistributedArrays.localpart{TS,N,P}(S::DStencil{TS,N,P})
    localS = localpart(S.v)
    Stencil{TS,N,P,typeof(localS)}(localS)
end
function DistributedArrays.localpart{S<:DistributedArrays.DArray}(m::Reservoirmodel{S})
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

function getresidual{T,N,P,S}(m, q, g::DGrid{T,N,P,S}, g_prev::DGrid{T,N,P,S})
    D = DistributedArrays.DArray(I -> getlocalresidual(localpart(m), localpart(q), localpart(g), localpart(g_prev)), g.A)
    return makegrid(D,P)
end
function getstencilArray{T,N,P,S}(m, q, g::DGrid{T,N,P,S}, g_prev::DGrid{T,N,P,S})
    D = DistributedArrays.DArray(I-> getstencilArray(localpart(m), localpart(q), localpart(g), localpart(g_prev)), g.A)
    return D
end
function getstencil{T,N,P,S}(m, q, g::DGrid{T,N,P,S}, g_prev::DGrid{T,N,P,S})
    SS = getstencilArray(m, q, g, g_prev)
    return Stencil{eltype(eltype(SS)),N,P,typeof(SS)}(SS)
end
