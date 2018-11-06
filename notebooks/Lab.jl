@everywhere using LinearAlgebra, StaticArrays, DistributedArrays
@everywhere include("../src/Reservoir.jl")
@everywhere using .Reservoir
using HDF5
porosity = h5read("../src/spe10data.h5", "data/porosity")
kx = h5read("../src/spe10data.h5", "data/kx")
ky = h5read("../src/spe10data.h5", "data/ky")
kz = h5read("../src/spe10data.h5", "data/kz")
kraw = [@SVector([kx[i,j,k], ky[i,j,k], kz[i,j,k]]) for i in 1:60, j in 1:220, k in 1:85];
Nx, Ny, Nz = 60, 220, 85;
@everywhere p_ref, ϕ_ref = 14.7, 0.2
@everywhere S_wc, S_or = 0.2, 0.2
@everywhere k_r_w(x)   = ((x-S_wc)/(1-S_wc-S_or))^2
@everywhere k_r_o(x)   = (1 - (x-S_wc)/(1-S_wc-S_or))^2
@everywhere p_cow(x)   = 6.3/log(0.00001)*log(x + 0.00001)
@everywhere C_water, C_r, C_oil    = 3e-6, 3e-6, 3e-6
@everywhere ρ_water(p) = 64.0*exp(C_water*(p-p_ref)) 
@everywhere ρ_oil(p)   = 53.0*exp(C_oil*(p-p_ref))   
@everywhere μ_water, μ_oil = 0.3, 0.3 # cp
saveper = (kx.+ky.+kz)./3;
## 3d model
Lx, Ly, Lz = 1200, 2200, 170
Δt = 0.5
Tf = 2000.0
Δx_d = distribute(fill(Lx/Nx, Nx, Ny, Nz))
Δy_d = distribute(fill(Ly/Ny, Nx, Ny, Nz), Δx_d)
Δz_d = distribute(fill(Lz/Nz, Nx, Ny, Nz), Δx_d)
z_d  = distribute(fill(12000.0, Nx, Ny, Nz), Δx_d)
ϕ_d = distribute(porosity, Δx_d)
k_d = makegrid(distribute(kraw),7)
model_d = Reservoirmodel(Δt, Tf, (Δx_d, Δy_d, Δz_d), z_d, k_d, p_ref, C_r, ϕ_ref, ϕ_d, 
                k_r_w, k_r_o, p_cow, C_water, C_oil, ρ_water, ρ_oil, μ_water, μ_oil);
## Porosity proportional control
Total = 5000.0*Δt
q = zeros(Nx, Ny, Nz, 2)
for i in (1,Nx), j in (1,Ny), kk in 1:Nz
    q[i,j,kk,2] = Total*(saveper[i,j,kk]/sum(saveper[i,j,:]))/4
end
halfx, halfy = round(Int, Nx/2),round(Int, Ny/2)
for i in 1:Nz
    q[halfx,halfy,i,1] = -Total*(saveper[halfx,halfy,i]/sum(saveper[halfx,halfy,:]))
end 
q_d = distribute(q);
g_guess_d  = ((makegrid(distribute(fill(6000.0,Nx,Ny,Nz)),7),makegrid(distribute(fill(0.2,Nx,Ny,Nz)),7)));
ReservoirSolve(model_d, q_d, g_guess_d, 1; tol_relnorm=0.99); ##Compiling