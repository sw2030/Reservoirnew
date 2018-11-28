
struct Reservoirmodel{S<:AbstractArray}
    q_oil::S
    q_water::S
    Δ::NTuple{3,S}
    z::S
    k::Grid
    p_ref::Float64
    C_r::Float64
    ϕ_ref::Float64
    ϕ::S
    k_r_w::Function
    k_r_o::Function
    p_cow::Function
    C_w::Float64
    C_o::Float64
    ρ_w::Function
    ρ_o::Function
    μ_w::Float64
    μ_o::Float64
end
Base.size(M::Reservoirmodel) = size(M.z)

### Need update for 2D cases!!
#=
function res_f(m, q, g, g_prev, i, j)

    Nx, Ny = size(m.Δ[1])

    # Production
    q_water = q[1]
    q_oil   = q[2]

    im1 = max(1, i-1)
    ip1 = min(Nx, i+1)
    jm1 = max(1, j-1)
    jp1 = min(Ny, j+1)

    # The zy area between two blocks is calculated as the arithmetic mean of the zy area at each block center
    # See note 3
    A_west  = (m.Δ[3][im1, j]*m.Δ[2][im1, j] + m.Δ[3][i, j]*m.Δ[2][i, j])/2
    A_east  = (m.Δ[3][ip1, j]*m.Δ[2][ip1, j] + m.Δ[3][i, j]*m.Δ[2][i, j])/2
    A_south = (m.Δ[3][i, jm1]*m.Δ[1][i, jm1] + m.Δ[3][i, j]*m.Δ[1][i, j])/2
    A_north = (m.Δ[3][i, jp1]*m.Δ[1][i, jp1] + m.Δ[3][i, j]*m.Δ[1][i, j])/2

        # The interface permeability is the harmonic average of the two grid blocks
    # See note 3
    k_west  = (m.Δ[1][im1, j]+m.Δ[1][i, j])*m.k[i-1, j]*m.k[i, j]/(m.Δ[1][im1, j]*m.k[i, j] + m.Δ[1][i, j]*m.k[i-1, j])
    k_east  = (m.Δ[1][ip1, j]+m.Δ[1][i, j])*m.k[i+1, j]*m.k[i, j]/(m.Δ[1][ip1, j]*m.k[i, j] + m.Δ[1][i, j]*m.k[i+1, j])
    k_south = (m.Δ[2][i, jm1]+m.Δ[2][i, j])*m.k[i, j-1]*m.k[i, j]/(m.Δ[2][i, jm1]*m.k[i, j] + m.Δ[2][i, j]*m.k[i, j-1])
    k_north = (m.Δ[2][i, jp1]+m.Δ[2][i, j])*m.k[i, j+1]*m.k[i, j]/(m.Δ[2][i, jp1]*m.k[i, j] + m.Δ[2][i, j]*m.k[i, j+1])

    S_water_im1j   = g[1,2]
    S_water_ijm1   = g[2,2]
    S_water_ij     = g[3,2]
    S_water_ijp1   = g[4,2]
    S_water_ip1j   = g[5,2]
    S_water_prev   = g_prev[i, j][2]

    S_oil_im1j   = 1-S_water_im1j
    S_oil_ijm1   = 1-S_water_ijm1
    S_oil_ij     = 1-S_water_ij
    S_oil_ijp1   = 1-S_water_ijp1
    S_oil_ip1j   = 1-S_water_ip1j
    S_oil_prev   = 1-S_water_prev

    # Pressure
    p_oil_im1j   = g[1,1]
    p_oil_ijm1   = g[2,1]
    p_oil_ij     = g[3,1]
    p_oil_ijp1   = g[4,1]
    p_oil_ip1j   = g[5,1]
    p_oil_prev   = g_prev[i, j][1]

    p_water_im1j = p_oil_im1j - m.p_cow(S_water_im1j)
    p_water_ijm1 = p_oil_ijm1 - m.p_cow(S_water_ijm1)
    p_water_ij   = p_oil_ij   - m.p_cow(S_water_ij)
    p_water_ijp1 = p_oil_ijp1 - m.p_cow(S_water_ijp1)
    p_water_ip1j = p_oil_ip1j - m.p_cow(S_water_ip1j)
    p_water_prev = p_oil_prev - m.p_cow(S_water_prev)


    # 5.615 is oil field units correction factor. See Note 4.
    V_ij      = m.Δ[1][i,j]*m.Δ[2][i,j]*m.Δ[3][i,j]*m.ϕ[i,j]/5.615
    V_ij_prev = m.Δ[1][i,j]*m.Δ[2][i,j]*m.Δ[3][i,j]*m.ϕ[i,j]/5.615

    # Fluid potentials
    Φ_water_im1j = p_water_im1j - m.ρ_w(p_water_im1j)*m.z[im1, j]/144.0
    Φ_water_ijm1 = p_water_ijm1 - m.ρ_w(p_water_ijm1)*m.z[i, jm1]/144.0
    Φ_water_ij   = p_water_ij   - m.ρ_w(p_water_ij)  *m.z[i, j]  /144.0
    Φ_water_ijp1 = p_water_ijp1 - m.ρ_w(p_water_ijp1)*m.z[i, jp1]/144.0
    Φ_water_ip1j = p_water_ip1j - m.ρ_w(p_water_ip1j)*m.z[ip1, j]/144.0

    Φ_oil_im1j = p_oil_im1j - m.ρ_o(p_oil_im1j)*m.z[im1, j]/144.0
    Φ_oil_ijm1 = p_oil_ijm1 - m.ρ_o(p_oil_ijm1)*m.z[i, jm1]/144.0
    Φ_oil_ij   = p_oil_ij   - m.ρ_o(p_oil_ij)  *m.z[i, j]  /144.0
    Φ_oil_ijp1 = p_oil_ijp1 - m.ρ_o(p_oil_ijp1)*m.z[i, jp1]/144.0
    Φ_oil_ip1j = p_oil_ip1j - m.ρ_o(p_oil_ip1j)*m.z[ip1, j]/144.0

    # Upstream condition
    k_r_water_west  = Φ_water_im1j > Φ_water_ij ? m.k_r_w(S_water_im1j)*m.ρ_w(p_water_im1j) : m.k_r_w(S_water_ij)*m.ρ_w(p_water_ij)
    k_r_water_east  = Φ_water_ip1j > Φ_water_ij ? m.k_r_w(S_water_ip1j)*m.ρ_w(p_water_ip1j) : m.k_r_w(S_water_ij)*m.ρ_w(p_water_ij)
    k_r_water_south = Φ_water_ijm1 > Φ_water_ij ? m.k_r_w(S_water_ijm1)*m.ρ_w(p_water_ijm1) : m.k_r_w(S_water_ij)*m.ρ_w(p_water_ij)
    k_r_water_north = Φ_water_ijp1 > Φ_water_ij ? m.k_r_w(S_water_ijp1)*m.ρ_w(p_water_ijp1) : m.k_r_w(S_water_ij)*m.ρ_w(p_water_ij)

    k_r_oil_west  = Φ_oil_im1j > Φ_oil_ij ? m.k_r_o(S_water_im1j)*m.ρ_o(p_oil_im1j) : m.k_r_o(S_water_ij)*m.ρ_o(p_oil_ij)
    k_r_oil_east  = Φ_oil_ip1j > Φ_oil_ij ? m.k_r_o(S_water_ip1j)*m.ρ_o(p_oil_ip1j) : m.k_r_o(S_water_ij)*m.ρ_o(p_oil_ij)
    k_r_oil_south = Φ_oil_ijm1 > Φ_oil_ij ? m.k_r_o(S_water_ijm1)*m.ρ_o(p_oil_ijm1) : m.k_r_o(S_water_ij)*m.ρ_o(p_oil_ij)
    k_r_oil_north = Φ_oil_ijp1 > Φ_oil_ij ? m.k_r_o(S_water_ijp1)*m.ρ_o(p_oil_ijp1) : m.k_r_o(S_water_ij)*m.ρ_o(p_oil_ij)

    Δx_west  = (m.Δ[1][im1, j] + m.Δ[1][i, j])/2
    Δx_east  = (m.Δ[1][ip1, j] + m.Δ[1][i, j])/2

    Δy_south = (m.Δ[2][i, jm1] + m.Δ[2][i, j])/2
    Δy_north = (m.Δ[2][i, jp1] + m.Δ[2][i, j])/2

    # The 1.127e-3 factor is oil field units. See Note 4.
    #T_water_west  = i == 1    ? 0.0 : 1.127e-3*k_west*k_r_water_west/m.μ_w*A_west/Δx_west # boundary condition
    #T_water_east  = i == Nx   ? 0.0 : 1.127e-3*k_east*k_r_water_east/m.μ_w*A_east/Δx_east
    #T_water_south = j == 1    ? 0.0 : 1.127e-3*k_south*k_r_water_south/m.μ_w*A_south/Δx_south
    #T_water_north = j == Ny   ? 0.0 : 1.127e-3*k_north*k_r_water_north/m.μ_w*A_north/Δx_north

    #T_oil_west    = i == 1    ? 0.0 : 1.127e-3*k_west*k_r_oil_west/m.μ_o*A_west/Δx_west
    #T_oil_east    = i == Nx   ? 0.0 : 1.127e-3*k_east*k_r_oil_east/m.μ_o*A_east/Δx_east
    #T_oil_south   = j == 1    ? 0.0 : 1.127e-3*k_south*k_r_oil_south/m.μ_o*A_south/Δx_south
    #T_oil_north   = j == Ny   ? 0.0 : 1.127e-3*k_north*k_r_oil_north/m.μ_o*A_north/Δx_north

    T_water_west  = 1.127e-3*k_west*k_r_water_west/m.μ_w*A_west/Δx_west # boundary condition
    T_water_east  = 1.127e-3*k_east*k_r_water_east/m.μ_w*A_east/Δx_east
    T_water_south = 1.127e-3*k_south*k_r_water_south/m.μ_w*A_south/Δy_south
    T_water_north = 1.127e-3*k_north*k_r_water_north/m.μ_w*A_north/Δy_north

    T_oil_west    = 1.127e-3*k_west*k_r_oil_west/m.μ_o*A_west/Δx_west
    T_oil_east    = 1.127e-3*k_east*k_r_oil_east/m.μ_o*A_east/Δx_east
    T_oil_south   = 1.127e-3*k_south*k_r_oil_south/m.μ_o*A_south/Δy_south
    T_oil_north   = 1.127e-3*k_north*k_r_oil_north/m.μ_o*A_north/Δy_north

    residual_water_ij = T_water_west*(Φ_water_im1j - Φ_water_ij)  +
                       T_water_east*(Φ_water_ip1j - Φ_water_ij)   +
                       T_water_south*(Φ_water_ijm1 - Φ_water_ij)  +
                       T_water_north*(Φ_water_ijp1 - Φ_water_ij)  -
                       q_water*m.ρ_w(p_water_ij)                  - # production
                       (V_ij*S_water_ij*m.ρ_w(p_water_ij)         -
                       V_ij_prev*S_water_prev*m.ρ_w(p_water_prev))/m.Δt

    residual_oil_ij   = T_oil_west*(Φ_oil_im1j - Φ_oil_ij)       +
                       T_oil_east*(Φ_oil_ip1j - Φ_oil_ij)        +
                       T_oil_south*(Φ_oil_ijm1 - Φ_oil_ij)       +
                       T_oil_north*(Φ_oil_ijp1 - Φ_oil_ij)       -
                       q_oil*m.ρ_o(p_oil_ij)                     - # production
                       (V_ij*S_oil_ij*m.ρ_o(p_oil_ij)            -
                       V_ij_prev*S_oil_prev*m.ρ_o(p_oil_prev))/m.Δt

    return [residual_water_ij, residual_oil_ij]
end

function res_each{T}(m, q, g::Grid{T,2,5,Matrix{T}}, g_prev::Grid{T,2,5,Matrix{T}}, i, j)
    return res_f(m, q[i,j,:], [g[i-1,j][1] g[i-1,j][2]; g[i,j-1][1] g[i,j-1][2]; g[i,j][1] g[i,j][2];
                  g[i,j+1][1] g[i,j+1][2]; g[i+1,j][1] g[i+1,j][2]], g_prev, i, j)
end
function getresidual{T}(m, q, g::Grid{T,2,5,Matrix{T}}, g_prev::Grid{T,2,5,Matrix{T}})
    Nx, Ny = size(m.Δ[1])
    return makegrid([SVector{2,Float64}(res_each(m, q, g, g_prev, i, j)) for i = 1:Nx, j = 1:Ny], 5)
end
## For distributed cases, return non-grid data
function getlocalresidual{T}(m, q, g::Grid{T,2,5,Matrix{T}}, g_prev::Grid{T,2,5,Matrix{T}})
    Nx, Ny = size(m.Δ[1])
    return [SVector{2,Float64}(res_each(m, q, g, g_prev, i, j)) for i = 1:Nx, j = 1:Ny]
end
function getstencilArray{T}(m, q, g::Grid{T,2,5,Matrix{T}}, g_prev::Grid{T,2,5,Matrix{T}})
    Nx, Ny = size(m.Δ[1])
    stencilArray = Array{StencilPoint{SMatrix{2,2,Float64,4},2,5},2}(Nx, Ny)
    for i in 1:Nx, j in 1:Ny
        J = ForwardDiff.jacobian(θ -> res_f(m, q[i,j,:], θ, g_prev, i, j), [g[i-1,j][1] g[i-1,j][2]; g[i,j-1][1] g[i,j-1][2];
                                                      g[i,j][1] g[i,j][2]; g[i,j+1][1] g[i,j+1][2]; g[i+1,j][1] g[i+1,j][2]])
        stencilArray[i,j] = StencilPoint{SMatrix{2,2,Float64,4},2,5}(
            (@SMatrix([J[1,1] J[1,6]; J[2,1] J[2,6]]),
             @SMatrix([J[1,2] J[1,7]; J[2,2] J[2,7]]),
             @SMatrix([J[1,3] J[1,8]; J[2,3] J[2,8]]),
             @SMatrix([J[1,4] J[1,9]; J[2,4] J[2,9]]),
             @SMatrix([J[1,5] J[1,10]; J[2,5] J[2,10]])))
    end
    return stencilArray
end=#
#=
function ReservoirSolve(m, q, g_guess, n_step ; tol_relnorm=1e-2, tol_dgnorm=1.0, tol_resnorm=10.0, tol_gmres=1e-4, n_restart=20, n_iter=50)
    psgrid_old = copy(g_guess)
    psgrid_new, result = copy(psgrid_old), Any[]
    for steps in 1:n_step
        RES = getresidual(m, q, psgrid_new, psgrid_old)
        norm_RES_save, norm_dg = norm(RES), 10000.0
        norm_RES = norm_RES_save
        println("\nstep ", steps, "  norm_RES : ", norm_RES)
        while(norm_RES/norm_RES_save > tol_relnorm && norm_dg > tol_dgnorm && norm_RES > tol_resnorm)
            JAC = getstencil(m, q, psgrid_new, psgrid_old)
            precP, precE = Reservoir.make_P_E_precond_1(JAC)
            print("GMRES start... ")
            gmresresult = stencilgmres(JAC, RES, n_restart; tol=tol_gmres, maxiter=n_iter, M=(t->Reservoir.precond_1(precP,precE,t)), ifprint=true)
            println(" ...GMRES done")
            LinearAlgebra.axpy!(-1.0, gmresresult[1], psgrid_new)
            RES = getresidual(m, q, psgrid_new, psgrid_old)
            norm_RES, norm_dg = norm(RES), norm(gmresresult[1])
            @show norm_RES, norm_dg
        end
        copyto!(psgrid_old, psgrid_new)
        push!(result, norm(psgrid_old[1]))
        push!(result, norm(psgrid_old[2]))
    end
    print("\nSolve done")
    return result, psgrid_new
end
=#
function getstencil(m, Δt, g::Grid{T,N,P,S}, g_prev::Grid{T,N,P,S}) where {T,N,P,S}
    SS = getstencilArray(m, Δt, g, g_prev)
    return Stencil{eltype(eltype(SS)),N,P,typeof(SS)}(SS)
end
function getstencil(m, Δt, g::MGrid{M,T,N,P,S}, g_prev::MGrid{M,T,N,P,S}) where {M,T,N,P,S}
    return MStencil{M*M,T,N,P,Array{StencilPoint{T,N,P},N}}(Stencil{T,N,P,Array{StencilPoint{T,N,P},N}}.(getstencilArray(m, Δt, g, g_prev)))
end
