addprocs(4);
using DistributedArrays
@everywhere using DistributedArrays
using ForwardDiff
@everywhere using ForwardDiff
using StaticArrays
@everywhere using StaticArrays
@everywhere include("Stencil.jl")
@everywhere include("stencilgmres.jl")
@everywhere include("reservoirfunc.jl")
@everywhere include("distributed.jl")
@everywhere include("res3d.jl")
using BenchmarkTools
permdata = open("spe_perm.dat")
phidata  = open("spe_phi.dat")
raw1 = readdlm(phidata)
raw2 = readdlm(permdata)
close(permdata)
close(phidata)
porosity = reshape(raw1', 60, 220, 85)
lineperm = reshape(raw2', 3366000, 1)
kx = reshape(lineperm[1:1122000]', 60, 220, 85)
ky = reshape(lineperm[1122001:2244000]', 60, 220, 85)
kz = reshape(lineperm[2244001:end]', 60, 220, 85)
savepor = copy(porosity)
saveper = (kx.+ky.+kz)./3
## Porosity pre-processing
for id in eachindex(porosity)
    if abs(porosity[id])<1e-3
        porosity[id] = 0.2
        kx[id], ky[id], kz[id] = 1e-10, 1e-10, 1e-10
    end
end
kraw = [SVector{3,Float64}([kx[i,j,k], ky[i,j,k], kz[i,j,k]]) for i in 1:60, j in 1:220, k in 1:85];
Nx = 10
Ny = 10
Nz = 10
offset = 1;
@everywhere p_ref = 14.7
@everywhere C_r   = 3e-6
@everywhere ϕ_ref = 0.2
#=everywhere=# ϕ     = porosity[1:Nx, 1:Ny, 1:Nz]
@everywhere S_wc, S_or = 0.2, 0.2
@everywhere k_r_w(x)   = ((x-S_wc)/(1-S_wc-S_or))^2
@everywhere k_r_o(x)   = (1 - (x-S_wc)/(1-S_wc-S_or))^2
@everywhere p_cow(x)   = 0
@everywhere C_water    = 3e-6
@everywhere C_oil      = 3e-6
@everywhere ρ_water(p) = 64.0*exp(C_water*(p-p_ref))
@everywhere ρ_oil(p)   = 53.0*exp(C_oil*(p-p_ref))
@everywhere μ_water = 0.3 # cp
@everywhere μ_oil   = 3.0 # cp
## 3d model
Lx = 1200
Ly = 2200
Lz = 170
Δt = 0.01
Tf = 2000.0
Δx = (fill(Lx/Nx, Nx, Ny, Nz))
Δy = (fill(Ly/Ny, Nx, Ny, Nz))
Δz = fill(Lz/Nz, Nx, Ny, Nz)
z  = fill(12000.0, Nx, Ny, Nz)
k = makegrid(kraw[1:Nx,1:Ny,1:Nz],7)
Δx_d = distribute(Δx)
Δy_d = distribute(Δy,Δx_d)
Δz_d = distribute(Δz,Δx_d)
z_d  = distribute(z,Δx_d)
ϕ_d  = distribute(ϕ,Δx_d)
k_d  = makegrid(distribute(kraw[1:Nx,1:Ny,1:Nz],Δx_d),7)
model = Reservoirmodel(Δt, Tf, (Δx, Δy, Δz), z, k, p_ref, C_r, ϕ_ref, ϕ,
                k_r_w, k_r_o, p_cow, C_water, C_oil, ρ_water, ρ_oil, μ_water, μ_oil)
model_d = Reservoirmodel(Δt, Tf, (Δx_d, Δy_d, Δz_d), z_d, k_d, p_ref, C_r, ϕ_ref, ϕ_d,
                k_r_w, k_r_o, p_cow, C_water, C_oil, ρ_water, ρ_oil, μ_water, μ_oil);
Total = 5000.0
q = zeros(Nx, Ny, Nz, 2)
for i in (1,Nx), j in (1,Ny), k in 1:Nz
    q[i,j,k,2] = Total*(saveper[i,j,k]/sum(saveper[i,j,:]))/4
end
halfx, halfy = round(Int, Nx/2),round(Int, Ny/2)
for k in 1:Nz
    q[halfx,halfy,k,1] = -Total*(saveper[halfx,halfy,k]/sum(saveper[halfx,halfy,:]))
end
q_d = distribute(q);

init_d     = distribute([SVector{2,Float64}([6000.0,0.2]) for i in 1:Nx, j in 1:Ny, k in 1:Nz])
testgrid_d = makegrid(init_d, 7)
g_guess_d = testgrid_d;
S_d = getstencil(model_d, q_d, g_guess_d, testgrid_d)
res_d = getresidual(model_d, q_d, g_guess_d, testgrid_d);
rescopy_d = getresidual(model_d, q_d, g_guess_d, testgrid_d);
init       = [SVector{2,Float64}([6000.0,0.2]) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
testgrid   = makegrid(init, 7);
g_guess   = testgrid;
S = getstencil(model, q, g_guess, testgrid)
res = getresidual(model, q, g_guess, testgrid);
copyres = getresidual(model, q, g_guess, testgrid);
N = Nx*Ny*Nz
Nxy = Nx*Ny
#=A = spzeros(2N,2N)
A2= spzeros(StaticArrays.SArray{Tuple{2,2},Float64,2,4},N,N)
b = zeros(2N)
b2 = zeros(StaticArrays.SArray{Tuple{2},Float64,1,2},N)
for i in 1:Nx, j in 1:Ny, k in 1:Nz
    nd = (k-1)*Nxy+(j-1)*Nx+i
    A[(2*nd-1):2*nd, (2*nd-1):2*nd] = S[i,j,k].value[4]
    A2[nd,nd] = S[i,j,k].value[4]
    if i!=1
        A[(2*nd-1):2*nd, (2*nd-3):(2*nd-2)]           = S[i,j,k].value[1]
        A2[nd, nd-1] = S[i,j,k].value[1]
    end
    if j!=1
        A[(2*nd-1):2*nd, (2*nd-2*Nx-1):(2*nd-2*Nx)]   = S[i,j,k].value[2]
        A2[nd,nd-Nx] = S[i,j,k].value[2]
    end
    if k!=1
        A[(2*nd-1):2*nd, (2*nd-2*Nxy-1):(2*nd-2*Nxy)] = S[i,j,k].value[3]
        A2[nd,nd-Nxy] = S[i,j,k].value[3]
    end
    if k!=Nz
        A[(2*nd-1):2*nd, (2*nd+2*Nxy-1):(2*nd+2*Nxy)] = S[i,j,k].value[5]
        A2[nd,nd+Nxy] = S[i,j,k].value[5]
    end
    if j!=Ny
        A[(2*nd-1):2*nd, (2*nd+2*Nx-1):(2*nd+2*Nx)]   = S[i,j,k].value[6]
        A2[nd,nd+Nx] = S[i,j,k].value[6]
    end
    if i!=Nx
        A[(2*nd-1):2*nd, (2*nd+1):(2*nd+2)]           = S[i,j,k].value[7]
        A2[nd,nd+1] = S[i,j,k].value[7]
    end
    b[(2*nd-1):(2*nd)] = res[i,j,k]
    b2[nd] = res[i,j,k]
    #if (j==1)&&(k==1) print(i) end
end
b2copy = copy(b2)
bcopy = copy(b);=#
function mybench(aa, bb, cc)
    @btime A_mul_B!(1.0, $aa, $bb, 2.0, $cc);
end
function mybench2(aa, bb)
    @btime stencilgmres($aa, $bb, 10;maxiter=5,ifprint=false,tol=1e-8);
end
#mybench(S_d, res_d, rescopy_d)
#mybench(S, res, copyres);
#mybench(A, b, bcopy);
#mybench(A2, b2, b2copy);
#@btime zero(res);
#@btime zero2(res_d);
#mybench2(S, res)
#mybench2(S_d, res_d)
print(typeof(res_d.A))
print(size(res_d.A))
