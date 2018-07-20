#addprocs(4);
using DistributedArrays
using ForwardDiff
using StaticArrays
include("Stencil.jl")
include("stencilgmres.jl")
include("reservoirfunc.jl")
include("distributed.jl")
include("res3d.jl")
include("preconditioning.jl")
using BenchmarkTools
function fullm{TS}(S::Stencil{TS,3,7}, G)
    nx, ny, nz = size(S)
    SS = spzeros(nx*ny*nz, nx*ny*nz)
    GG = zeros(nx*ny*nz)
    for i in 1:nx, j in 1:ny, k in 1:nz
        nd = (i-1)*ny*nz+(j-1)*nz+k
        Sv = S.v[i,j,k]
        SS[nd,nd] += Sv.value[4]
        GG[nd] += G[i,j,k]
        if i!=1  SS[nd,nd-ny*nz] = Sv.value[1] end
        if i!=nx SS[nd,nd+ny*nz] = Sv.value[7] end
        if j!=1  SS[nd,nd-nz] = Sv.value[2] end
        if j!=nx SS[nd,nd+nz] = Sv.value[6] end
        if k!=1  SS[nd,nd-1]  = Sv.value[3] end
        if k!=ny SS[nd,nd+1]  = Sv.value[5] end
    end
    return SS, GG
end
function test(N,numz)
    Sarray = Array{StencilPoint{Float64,3,7},3}(N, N, N);
    for i in 1:N, j in 1:N, k in 1:N
        spt = 1000.0*randn(7).*rand([zeros(numz);1.0],7)
        if i==1 spt[1] = 0.0 end
        if j==1 spt[2] = 0.0 end
        if k==1 spt[3] = 0.0 end
        if k==N spt[5] = 0.0 end
        if j==N spt[6] = 0.0 end
        if i==N spt[7] = 0.0 end
        sindex = Int64[]
        for m in 1:7
            if spt[m] != 0.0 push!(sindex,m) end
        end
        si = Tuple([sindex;zeros(7-length(sindex))])
        spt = Tuple(spt)
        Sarray[i,j,k] = StencilPoint{Float64,3,7}(spt,si,length(sindex))
    end
    S1 = Stencil{Float64,3,7,typeof(Sarray)}(Sarray);
    G1 = makegrid(randn(N,N,N),7)
    SA, GA = fullm(S1, G1);
    println(length(SA.nzval)/(N*N*N*7-6*N*N));
    return S1, G1, SA, GA;
end
