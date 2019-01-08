using Distributed, DistributedArrays, LinearAlgebra
import Base.close

const DGrid{T,N,P,S} = Grid{T,N,1,DArray{T,N,Grid{T,N,P,S}}}
const DMGrid{M,T,N,P,S} = MGrid{M,T,N,1,DArray{T,N,Grid{T,N,P,S}}}
const DStencil{T,N,P} = Stencil{T,N,P,<:DArray}
const DMStencil{MM,T,N,P} = MStencil{MM,T,N,P,<:DArray}

Distributed.procs(DG::DGrid)   = procs(DG.A)
Distributed.procs(DMG::DMGrid) = procs(DMG[1])
Base.close(G::DGrid) = Base.close(G.A)
Base.close(G::DMGrid) = Base.close.(G)
Base.close(S::DStencil) = Base.close(S.v)
Base.close(S::DMStencil) = Base.close.(S.stencils)


###=====================================================================
### Define Local parts. Should localpart of Stencils have type parameters?
###=====================================================================
DistributedArrays.localpart(G::DGrid) = localpart(G.A)
DistributedArrays.localpart(G::DMGrid) = map(localpart, G) 
DistributedArrays.localpart(S::DStencil) = Stencil(localpart(S.v))
DistributedArrays.localpart(S::DMStencil) = MStencil(map(localpart,S.stencils))
function DistributedArrays.localpart(m::Reservoirmodel{S}) where {S<:DistributedArrays.DArray}
    Reservoirmodel(
        localpart(m.q_oil),
        localpart(m.q_water),
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


###=====================================================================
### Base functions
###=====================================================================
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
function /(g::DGrid{T,N,P,S}, x::Float64) where {T,N,P,S}
    D = DistributedArrays.DArray(I-> /(localpart(g.A),x), g.A)
    Grid{T,N,1,typeof(D)}(D)
end
/(mg::DMGrid{M,T,N,P,S}, x::Float64) where {M,T,N,P,S} = mg./x
function *(x::Float64, g::DGrid{T,N,P,S}) where {T,N,P,S}
    D = DistributedArrays.DArray(I-> *(localpart(g.A),x), g.A)
    Grid{T,N,1,typeof(D)}(D)
end
function *(g::DGrid{T,N,P,S}, x::Float64) where {T,N,P,S}
    D = DistributedArrays.DArray(I-> *(x,localpart(g.A)), g.A)
    Grid{T,N,1,typeof(D)}(D)
end


###=====================================================================
### Linear Algebra Routines
###=====================================================================
## Until DistributedArrays.jl dot for Array is well-implemented
function LinearAlgebra.dot(D1::DArray{T}, D2::DArray{T})::float(eltype(T)) where {T}
    r = asyncmap(procs(D1)) do p
        remotecall_fetch(p) do
            LinearAlgebra.dot(localpart(D1), localpart(D2))
        end
    end
    return reduce(+, r)
end

# Full gemv! for DGrid, DMGrid
function gemv!(a::Number, S::DStencil{TS,N}, x::DGrid{Txy,N},  b::Number, y::DGrid{Txy,N}) where {TS,Txy,N}
    @sync for id in procs(y)
        @async remotecall_fetch((a,S,x,b,y)->(gemv!(a,localpart(S),localpart(x),b,localpart(y)); nothing),id,a,S,x,b,y)
    end
    @sync for id in procs(y)
        @async remotecall_fetch(gridsync, id, id, y.A)
    end
    return y
end
function gemv!(a::Number, MS::DMStencil{4,TS,3}, x::DMGrid{2,Txy,3},  b::Number, y::DMGrid{2,Txy,3}) where {Txy,TS}
    asyncmap(procs(y)) do p
        remotecall_fetch(p) do
            gemv!(a, localpart(MS[1]), localpart(x[1]), b, localpart(y[1]))
            gemv!(a, localpart(MS[2]), localpart(x[2]), 1, localpart(y[1]))
            gemv!(a, localpart(MS[3]), localpart(x[1]), b, localpart(y[2]))            
            gemv!(a, localpart(MS[4]), localpart(x[2]), 1, localpart(y[2]))
            return nothing
        end
    end
    @sync for id in procs(y)
        @async remotecall_fetch(id) do
            gridsync(id, y[1].A)
            gridsync(id, y[2].A)
        end
    end
end
function gemv2!(a::Number, MS::DMStencil{4,TS,3}, x::DMGrid{2,Txy,3},  b::Number, y::DMGrid{2,Txy,3}) where {Txy,TS}
    asyncmap(procs(y)) do p
        remotecall_fetch(p) do
            gemv!(a, localpart(MS[1]), localpart(x[1]), b, localpart(y[1]))
            gemv!(a, localpart(MS[2]), localpart(x[2]), 1, localpart(y[1]))
            gemv!(a, localpart(MS[3]), localpart(x[1]), b, localpart(y[2]))            
            gemv!(a, localpart(MS[4]), localpart(x[2]), 1, localpart(y[2]))
            return nothing
        end
    end
    @sync for id in procs(y)
        @async remotecall_fetch(id) do
            gridsync2(id, y[1].A)
            gridsync2(id, y[2].A)
        end
    end
end
### A*b version for DGrid, DMGrid
function mul!(y::DGrid{Txy,N,P}, S::DStencil{TS,N,P}, x::DGrid{Txy,N,P}) where {TS,Txy,N,P}
    @sync for id in procs(y)
        @async remotecall_fetch((y,S,x) -> (mul!(localpart(y), localpart(S), localpart(x)); nothing), id, y, S, x)
    end
    @sync for id in procs(y)
        @async remotecall_fetch(gridsync, id, id, y.A)
    end
    return y
end
function mul!(y::DMGrid{2,Txy,3}, MS::DMStencil{4,TS,3}, x::DMGrid{2,Txy,3}) where {TS,Txy}
    mul!(y[1],MS[1],x[1])
    mul!(y[1],MS[2],x[2])
    mul!(y[2],MS[3],x[1])
    mul!(y[2],MS[4],x[2])
    return y
end
Base.:*(S::DStencil{TS,N,P},x::DGrid{Txy,N,P}) where {TS,Txy,N,P}                   = mul!(zero(x),S,x)
Base.:*(S::DMStencil{MM,TS,N,P},x::DMGrid{M,Txy,N,P,AA}) where {MM,M,TS,Txy,N,P,AA} = mul!(zero(x),S,x)

## Other Linear Algebra Routines
LinearAlgebra.norm(g::DGrid) = norm(g.A)
LinearAlgebra.norm(g::DMGrid) = return norm(norm.(g))
LinearAlgebra.dot(gx::DGrid{T,N,P,S}, gy::DGrid{T,N,P,S}) where {T,N,P,S}            = LinearAlgebra.dot(gx.A, gy.A) ## This is okay because dot works on inner DArray by automatic inheritance
LinearAlgebra.dot(gx::DMGrid{M,T,N,P,S}, gy::DMGrid{M,T,N,P,S}) where {M,T,N,P,S}    = reduce(+, LinearAlgebra.dot.(gx, gy))

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
function LinearAlgebra.axpy!(a::Number, g1::DGrid{T,N,P,S}, g2::DGrid{T,N,P,S}) where {T,N,P,S}
    @sync begin
        for id in procs(g1.A)
            @async remotecall_fetch(id) do
                LinearAlgebra.axpy!(a, localpart(g1.A), localpart(g2.A))
                nothing
            end
        end
    end
    return g2
end
zero(g::DGrid{T,N,P,S}) where {T,N,P,S} = Grid{T,N,1,DArray{T,N,Grid{T,N,P,S}}}(DArray(I->zero(localpart(g)), g.A))
zero(mg::DMGrid{M,T,N,P,S}) where {M,T,N,P,S}  = zero.(mg)


###=====================================================================
### Grid Constructor
###=====================================================================
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
    remotecall_fetch(id) do
        if i!=1 
            copyto!((@view (localpart(D).A)[1,:,:]), 
                remotecall_fetch(m->(@view (localpart(m).A)[end-1,:,:]), D.pids[LinearIndices(s)[i-1,j,k]],D))
        end
        if i!=s[1]
            copyto!((@view (localpart(D).A)[end,:,:]), 
                remotecall_fetch(m->(@view (localpart(m).A)[2,:,:]), D.pids[LinearIndices(s)[i+1,j,k]],D))
        end
        if j!=1
            copyto!((@view (localpart(D).A)[:,1,:]), 
                remotecall_fetch(m->(@view (localpart(m).A)[:,end-1,:]), D.pids[LinearIndices(s)[i,j-1,k]],D))
        end
        if j!=s[2]
            copyto!((@view (localpart(D).A)[:,end,:]), 
                remotecall_fetch(m->(@view (localpart(m).A)[:,2,:]), D.pids[LinearIndices(s)[i,j+1,k]],D))
        end
        if k!=1
            copyto!((@view (localpart(D).A)[:,:,1]), 
                remotecall_fetch(m->(@view (localpart(m).A)[:,:,end-1]), D.pids[LinearIndices(s)[i,j,k-1]],D))
        end
        if k!=s[3]
            copyto!((@view (localpart(D).A)[:,:,end]), 
                remotecall_fetch(m->(@view (localpart(m).A)[:,:,2]), D.pids[LinearIndices(s)[i,j,k+1]],D))
        end
    end
    return nothing
end
function gridsync2(id, D::DistributedArrays.DArray{T,3}) where {T}
    s = size(D.pids)
    i,j,k = Tuple(CartesianIndices(s)[findfirst(isequal(id),D.pids)])
    remotecall_fetch(id) do
        if i!=1 
            copyto!((@view (localpart(D))[1,:,:]), 
                (@view D[last(D.indices[i-1,j,k][1]),D.indices[i-1,j,k][2],D.indices[i-1,j,k][3]]))
        end
        if i!=s[1]
            copyto!((@view (localpart(D))[end,:,:]), 
                (@view D[first(D.indices[i+1,j,k][1]),D.indices[i+1,j,k][2],D.indices[i+1,j,k][3]]))
        end
        if j!=1
            copyto!((@view (localpart(D))[:,1,:]), 
                (@view D[D.indices[i,j-1,k][1],last(D.indices[i,j-1,k][2]),D.indices[i,j-1,k][3]]))
        end
        if j!=s[2]
            copyto!((@view (localpart(D))[:,end,:]), 
                (@view D[D.indices[i,j+1,k][1],first(D.indices[i,j+1,k][2]),D.indices[i,j+1,k][3]]))
        end
        if k!=1
            copyto!((@view (localpart(D))[:,:,1]), 
                (@view D[D.indices[i,j,k-1][1],D.indices[i,j,k-1][2],last(D.indices[i,j,k-1][3])]))
        end
        if k!=s[3]
            copyto!((@view (localpart(D))[:,:,end]), 
                (@view D[D.indices[i,j,k+1][1],D.indices[i,j,k+1][2],first(D.indices[i,j,k+1][3])]))
        end
    end
    return nothing
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



###=====================================================================
### Compute Residual & Jacobian
###=====================================================================
function getresidual(m, Δt, g::DMGrid{M,T,N,P,S}, g_prev::DMGrid{M,T,N,P,S}) where {M,T,N,P,S}
    D = DistributedArrays.DArray(I-> getlocalresidual(localpart(m), Δt, localpart(g), localpart(g_prev)), m.z)
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
    close(D)
    return (Grid{T,N,1,typeof(D1)}(D1),Grid{T,N,1,typeof(D2)}(D2))
end
function getlocalstencil(m, Δt, g::MGrid{M,T,N,P,S}, g_prev::MGrid{M,T,N,P,S}) where {M,T,N,P,S}
    Nx, Ny, Nz = size(m)
    stencilArray = Array{NTuple{4,StencilPoint{Float64,3,7}},3}(undef,Nx, Ny, Nz)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        J = ForwardDiff.jacobian(θ -> res_f(m, Δt, θ, g_prev, i, j, k), [g[i-1,j,k,1] g[i-1,j,k,2]; g[i,j-1,k,1] g[i,j-1,k,2];
                                                                                    g[i,j,k-1,1] g[i,j,k-1,2]; g[i,j,k,1] g[i,j,k,2];
                                                                                    g[i,j,k+1,1] g[i,j,k+1,2]; g[i,j+1,k,1] g[i,j+1,k,2]; g[i+1,j,k,1] g[i+1,j,k,2]])
        stencilArray[i,j,k] = (StencilPoint{Float64,3,7}((J[1,1],J[1,2],J[1,3],J[1,4],J[1,5],J[1,6],J[1,7])), StencilPoint{Float64,3,7}((J[1,8],J[1,9],J[1,10],J[1,11],J[1,12],J[1,13],J[1,14])),
                    StencilPoint{Float64,3,7}((J[2,1],J[2,2],J[2,3],J[2,4],J[2,5],J[2,6],J[2,7])), StencilPoint{Float64,3,7}((J[2,8],J[2,9],J[2,10],J[2,11],J[2,12],J[2,13],J[2,14])))
    end
    return stencilArray
end
function getstencil(m, Δt, g::DMGrid{M,T,N,P,S}, g_prev::DMGrid{M,T,N,P,S}) where {M,T,N,P,S}
    D = DArray(I->getlocalstencil(localpart(m), Δt, localpart(g), localpart(g_prev)), m.z)
    D1 = DArray(I->map(t->t[1],localpart(D)), D)
    D2 = DArray(I->map(t->t[2],localpart(D)), D)
    D3 = DArray(I->map(t->t[3],localpart(D)), D)
    D4 = DArray(I->map(t->t[4],localpart(D)), D)
    close(D)
    return MStencil((Stencil{T,N,P,typeof(D1)}(D1), Stencil{T,N,P,typeof(D1)}(D2), Stencil{T,N,P,typeof(D1)}(D3), Stencil{T,N,P,typeof(D1)}(D4)))
end


###=====================================================================
### Preconditioning
###=====================================================================
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
function precond_1(Pinv::DMStencil{4,TS,3},E::DMStencil{4,TS,3},x::DMGrid{2,Tx,3}) where {TS,Tx}    
    result = copy(x)
    tmp = Pinv*x
    gemv!(-1, E, tmp, 1, result)
    gemv!(1, Pinv, result, 0, tmp)
    close(result)
    return tmp
end
function precond_2(Pinv::DMStencil{4,TS,3},E::DMStencil{4,TS,3},x::DMGrid{2,Tx,3}) where {TS,Tx} 
    tmp1 = copy(x)
    tmp2 = Pinv*x
    gemv!(-1, E, tmp2, 1, tmp1)   # tmp1 = x - EPx
    gemv!(1, Pinv, tmp1, 0, tmp2) # tmp2 = P(x-EPx)
    copyto!(tmp1, x)              # tmp1 = x
    gemv!(-1, E, tmp2, 1, tmp1)   # tmp1 = x - E(P(x-EPx))
    gemv!(1, Pinv, tmp1, 0, tmp2) # tmp2 = P(x-E(P(x-EPx)))
    close(tmp1)
    return tmp2
end

function stencilgmres(A, b, restrt::Int64; tol::Real=1e-5, maxiter::Int=200, ifprint=false, M=identity, x = zero(b))
    realn, bnrm2 = gridsize(b), norm(b)
    if bnrm2==0 bnrm2 = 1.0 end
    r = copy(b)
    gemv!(-1, A, x, 1, r)
    tmp = M(r)
    copyto!(r, tmp)
    if M!=identity close(tmp) end
    #err = norm(r)/bnrm2
    bnrm2 = norm(r)
    err = 1.0
    itersave = 0
    ismax = false
    errlog = Float64[]

    if err<tol
        copyto!(x, r)
        close(r)
        return x, ismax, itersave, err, errlog 
    end

    restrt=min(restrt, realn-1)
    Q = [zero(b) for i in 1:restrt+1]
    H = zeros(restrt+1, restrt)
    cs = zeros(restrt)
    sn = zeros(restrt)
    s = zeros(restrt+1)
    flag = -1
    isave = 1

    y = zeros(restrt+1)

    for iter in 1:maxiter
        push!(errlog, err)
        itersave = iter
        if ifprint print(".") end
        r = Q[1]
        copyto!(r, b)
        gemv!(-1, A, x, 1, r)
        tmp = M(r)
        copyto!(r, tmp)
        if M!=identity close(tmp) end
        fill!(s, 0.0)
        s[1] = norm(r)
        rmul!(r, inv(s[1]))

        for i in 1:restrt
            isave = i
            w = Q[i+1]
            gemv!(1, A, Q[i], 0, w)
            tmp = M(w)
            copyto!(w, tmp)
            if M!=identity close(tmp) end
            for k in 1:i
                H[k,i] = dot(w, Q[k])
                LinearAlgebra.axpy!(-H[k,i],Q[k],w)
            end
            H[i+1,i] = norm(w)
            rmul!(w, inv(H[i+1,i]))
            for k in 1:i-1
                temp     =  cs[k]*H[k,i] + sn[k]*H[k+1,i]
                H[k+1,i] = -sn[k]*H[k,i] + cs[k]*H[k+1,i]
                H[k,i]   = temp
            end

            cs[i], sn[i] = LinearAlgebra.givensAlgorithm(H[i, i], H[i+1, i])
            s[i+1] = -sn[i]*s[i]
            s[i]   = cs[i]*s[i]
            H[i,i] = cs[i]*H[i,i] + sn[i]*H[i+1,i]
            H[i+1,i] = 0.0
            err  = abs(s[i+1])/bnrm2
            
            if err < tol
                copyto!(y, s)
                ldiv!(UpperTriangular(view(H, 1:i, 1:i)), view(y, 1:i))
                for k in 1:i
                    LinearAlgebra.axpy!(y[k],Q[k],x)
                end
                flag = 0; break
            end
        end
        if  err < tol
            flag = 0
            break
        end
        copyto!(y, s)
        ldiv!(UpperTriangular(view(H, 1:restrt, 1:restrt)), view(y, 1:restrt))  
        for k in 1:restrt
            LinearAlgebra.axpy!(y[k],Q[k],x)  
        end
        copyto!(r, b)
        gemv!(-1, A, x, 1, r)
        tmp = M(r)
        copyto!(r, tmp)
        if M!=identity close(tmp) end
        s[isave+1] = norm(r)
        err = s[isave+1]/bnrm2
        if err<=tol
            flag = 0
            break
        end
    end
    if flag==-1
        ifprint==true && print(" Maxiter")
        ismax = true
    end
    close.(Q), close(r)

    return x, ismax, itersave, err, errlog
end
function stencilgmres2(A::DMStencil, b, restrt::Int64; tol::Real=1e-5, maxiter::Int=200, ifprint=false, M=identity, x = zero(b))
    realn, bnrm2 = gridsize(b), norm(b)
    if bnrm2==0 bnrm2 = 1.0 end
    r = copy(b)
    tmp = M(x)
    gemv!(-1, A, tmp, 1, r)
    if M!=identity close(tmp) end
    err = 1.0
    itersave = 0
    ismax = false
    errlog = Float64[]

    restrt=min(restrt, realn-1)
    Q = [zero(b) for i in 1:restrt+1]
    H = zeros(restrt+1, restrt)
    cs = zeros(restrt)
    sn = zeros(restrt)
    s = zeros(restrt+1)
    flag = -1
    isave = 1

    y = zeros(restrt+1)

    for iter in 1:maxiter
        push!(errlog, err)
        itersave = iter
        if ifprint&&(iter%10==0) print(".") end
        r = Q[1]
        copyto!(r, b)
        tmp = M(x)
        gemv!(-1, A, tmp, 1, r)
        if M!=identity close(tmp) end
        fill!(s, 0.0)
        s[1] = norm(r)
        rmul!(r, inv(s[1]))

        for i in 1:restrt
            isave = i
            w = Q[i+1]
            tmp = M(Q[i])
            gemv!(1, A, tmp, 0, w)
            if M!=identity close(tmp) end
            for k in 1:i
                H[k,i] = dot(w, Q[k])
                LinearAlgebra.axpy!(-H[k,i],Q[k],w)
            end
            H[i+1,i] = norm(w)
            rmul!(w, inv(H[i+1,i]))
            for k in 1:i-1
                temp     =  cs[k]*H[k,i] + sn[k]*H[k+1,i]
                H[k+1,i] = -sn[k]*H[k,i] + cs[k]*H[k+1,i]
                H[k,i]   = temp
            end

            cs[i], sn[i] = LinearAlgebra.givensAlgorithm(H[i, i], H[i+1, i])
            s[i+1] = -sn[i]*s[i]
            s[i]   = cs[i]*s[i]
            H[i,i] = cs[i]*H[i,i] + sn[i]*H[i+1,i]
            H[i+1,i] = 0.0
            err  = abs(s[i+1])/bnrm2
            
            
            if err < tol
                copyto!(y, s)
                ldiv!(UpperTriangular(view(H, 1:i, 1:i)), view(y, 1:i))
                for k in 1:i
                    LinearAlgebra.axpy!(y[k],Q[k],x)
                end
                flag = 0; break
            end
            
        end
        if  err < tol
            flag = 0
            break
        end
        copyto!(y, s)
        ldiv!(UpperTriangular(view(H, 1:restrt, 1:restrt)), view(y, 1:restrt))  
        for k in 1:restrt
            LinearAlgebra.axpy!(y[k],Q[k],x)  
        end
        copyto!(r, b)
        tmp = M(x)
        gemv!(-1, A, tmp, 1, r)
        if M!=identity close(tmp) end
        s[isave+1] = norm(r)
        err = s[isave+1]/bnrm2
        if err<=tol
            flag = 0
            break
        end
    end
    if flag==-1
        ifprint==true && print(" Maxiter")
        ismax = true
    end
    tmp = M(x)
    copyto!(x, tmp)
    if M!=identity close(tmp) end
    close.(Q), close(r)

    return x, ismax, itersave, err, errlog
end

