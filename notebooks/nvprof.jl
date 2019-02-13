using Distributed, CuArrays, NVTX
using CUDAdrv

addprocs(8);
@everywhere using LinearAlgebra, StaticArrays, DistributedArrays, CuArrays
@everywhere include("../src/Reservoir.jl")
@everywhere using .Reservoir
using HDF5
porosity = h5read("../src/spe10data.h5", "data/porosity")
kx = h5read("../src/spe10data.h5", "data/kx")
ky = h5read("../src/spe10data.h5", "data/ky")
kz = h5read("../src/spe10data.h5", "data/kz")
kraw = [@SVector([kx[i,j,k], ky[i,j,k], kz[i,j,k]]) for i in 1:60, j in 1:220, k in 1:85];
Nx, Ny, Nz = 60, 220, 85;
@everywhere p_ref, ϕ_ref = 14.7, 0.2                           ## ϕ_ref is unused, p_ref used for ρ
@everywhere S_wc, S_or = 0.2, 0.2                              ## SPE10 Config
@everywhere k_r_w(x)   = ((x-S_wc)/(1-S_wc-S_or))^2            ## SPE10 Config
@everywhere k_r_o(x)   = (1 - (x-S_wc)/(1-S_wc-S_or))^2        ## SPE10 Config ## Negligible Capillary Pressure?
@everywhere p_cow(x)   = 0.0                                   ## 6.3/log(0.00001)*log(x + 0.00001)
@everywhere C_water, C_r, C_oil    = 3e-6, 1e-6, 1.4e-6        ## C_w, C_r given SPE10 Config, C_oil?
@everywhere ρ_water(p) = 64.0*exp(C_water*(p-p_ref))           ## Anyway \approx 64.0 - given in SPE10
@everywhere ρ_oil(p)   = 53.0*exp(C_oil*(p-p_ref))             ## Anyway \approx 53.0 - given in SPE10
@everywhere μ_water, μ_oil = 0.3, 3.0 # cp ### SPE10 Config gives water viscosity, also oil pvt table gives \approx 3
## 3d model
## Porosity proportional control (PI propotional)

Lx, Ly, Lz = 1200, 2200, 170
Δx_d = distribute(fill(Lx/Nx, Nx, Ny, Nz))
Δy_d = distribute(fill(Ly/Ny, Nx, Ny, Nz), Δx_d)
Δz_d = distribute(fill(Lz/Nz, Nx, Ny, Nz), Δx_d)
z_d = distribute([12000.0+2.0*k-1.0 for i in 1:60, j in 1:220, k in 1:85], Δx_d)
# Top layer(k=1) z is 12001, end 12169
ϕ_d = distribute(porosity, Δx_d)
k_d = makegrid(distribute(kraw, Δx_d),7)
Total = 5000.0
q_oil_s   = zeros(Nx, Ny, Nz)
q_water_s = zeros(Nx, Ny, Nz)
for i in (1,Nx), j in (1,Ny), kk in 1:Nz
    q_oil_s[i,j,kk] = 1.0;
end
halfx, halfy = round(Int, Nx/2),round(Int, Ny/2)
for i in 1:Nz ### Injector
    q_water_s[halfx,halfy,i] = -Total*(kx[halfx,halfy,i]/sum(kx[halfx,halfy,:]))
end
q_oil   = distribute(q_oil_s);
q_water = distribute(q_water_s);
m = Reservoirmodel(q_oil, q_water, (Δx_d, Δy_d, Δz_d), z_d, k_d, p_ref, C_r, ϕ_ref, ϕ_d,
                k_r_w, k_r_o, p_cow, C_water, C_oil, ρ_water, ρ_oil, μ_water, μ_oil);

g_guess  = ((makegrid(distribute(fill(6000.0,Nx,Ny,Nz)),7),makegrid(distribute(fill(0.2,Nx,Ny,Nz)),7)));
R = SPE10Solve(m, [0.0;0.005], g_guess; tol_relnorm=0.99, tol_gmres=0.99); ##Compiling
close(R[1])
dres = getresidual(m, 0.005, g_guess, g_guess)
dJ = getstencil(m, 0.005, g_guess, g_guess)
cres = Reservoir.DtoCu(dres)
cJ   = Reservoir.DtoCu(dJ)
cP, cE = Reservoir.make_P_E_precond_1(cJ)
pM(t) = Reservoir.precond_1(cP, cE, t)
import .Reservoir:stencilgmres2, innerloop1!, innerloop2!, innerloop3!, gemv!, MGrid, MStencil, gridsize

Reservoir.stencilgmres3(cJ, cres, 3;M=pM, maxiter=1)
NVTX.@activate begin
NVTX.mark("Run 1")
CUDAdrv.@profile CuArrays.@sync Reservoir.stencilgmres3(cJ, cres, 3;M=pM, maxiter=2)			   
NVTX.mark("Run 2")
CUDAdrv.@profile CuArrays.@sync Reservoir.stencilgmres3(cJ, cres, 3;M=pM, maxiter=2)			   
end
