function res_f(m, q, g, g_prev, i, j, k;bth=true, p_bth=4000.0)

    Nx, Ny, Nz = size(m.Δ[1])

    # Production
    q_water = q[1]
    q_oil   = bth? 0.0 : q[2]

    im1 = max(1, i-1)
    ip1 = min(Nx, i+1)
    jm1 = max(1, j-1)
    jp1 = min(Ny, j+1)
    km1 = max(1, k-1)
    kp1 = min(Nz, k+1)

    # The zy area between two blocks is calculated as the arithmetic mean of the zy area at each block center
    # See note 3
    A_west  = (m.Δ[3][im1,j,k]*m.Δ[2][im1,j,k] + m.Δ[3][i,j,k]*m.Δ[2][i,j,k])/2
    A_east  = (m.Δ[3][ip1,j,k]*m.Δ[2][ip1,j,k] + m.Δ[3][i,j,k]*m.Δ[2][i,j,k])/2
    A_south = (m.Δ[3][i,jm1,k]*m.Δ[1][i,jm1,k] + m.Δ[3][i,j,k]*m.Δ[1][i,j,k])/2
    A_north = (m.Δ[3][i,jp1,k]*m.Δ[1][i,jp1,k] + m.Δ[3][i,j,k]*m.Δ[1][i,j,k])/2
    A_above = (m.Δ[1][i,j,kp1]*m.Δ[2][i,j,kp1] + m.Δ[1][i,j,k]*m.Δ[2][i,j,k])/2
    A_below = (m.Δ[1][i,j,km1]*m.Δ[2][i,j,km1] + m.Δ[1][i,j,k]*m.Δ[2][i,j,k])/2

    # The interface permeability is the harmonic average of the two grid blocks
    # See note 3
    k_west  = (m.Δ[1][im1,j,k]+m.Δ[1][i,j,k])*m.k[i-1,j,k][1]*m.k[i,j,k][1]/(m.Δ[1][im1,j,k]*m.k[i,j,k][1] + m.Δ[1][i,j,k]*m.k[i-1,j,k][1])
    k_east  = (m.Δ[1][ip1,j,k]+m.Δ[1][i,j,k])*m.k[i+1,j,k][1]*m.k[i,j,k][1]/(m.Δ[1][ip1,j,k]*m.k[i,j,k][1] + m.Δ[1][i,j,k]*m.k[i+1,j,k][1])
    k_south = (m.Δ[2][i,jm1,k]+m.Δ[2][i,j,k])*m.k[i,j-1,k][2]*m.k[i,j,k][2]/(m.Δ[2][i,jm1,k]*m.k[i,j,k][2] + m.Δ[2][i,j,k]*m.k[i,j-1,k][2])
    k_north = (m.Δ[2][i,jp1,k]+m.Δ[2][i,j,k])*m.k[i,j+1,k][2]*m.k[i,j,k][2]/(m.Δ[2][i,jp1,k]*m.k[i,j,k][2] + m.Δ[2][i,j,k]*m.k[i,j+1,k][2])
    k_below = (m.Δ[3][i,j,km1]+m.Δ[3][i,j,k])*m.k[i,j,k-1][3]*m.k[i,j,k][3]/(m.Δ[3][i,j,km1]*m.k[i,j,k][3] + m.Δ[3][i,j,k]*m.k[i,j,k-1][3])
    k_above = (m.Δ[3][i,j,kp1]+m.Δ[3][i,j,k])*m.k[i,j,k+1][3]*m.k[i,j,k][3]/(m.Δ[3][i,j,kp1]*m.k[i,j,k][3] + m.Δ[3][i,j,k]*m.k[i,j,k+1][3])

    S_water_im1jk   = g[1,2]
    S_water_ijm1k   = g[2,2]
    S_water_ijkm1   = g[3,2]
    S_water_ijk     = g[4,2]
    S_water_ijkp1   = g[5,2]
    S_water_ijp1k   = g[6,2]
    S_water_ip1jk   = g[7,2]
    S_water_prev   = g_prev[i, j, k][2]

    S_oil_im1jk   = 1-S_water_im1jk
    S_oil_ijm1k   = 1-S_water_ijm1k
    S_oil_ijkm1   = 1-S_water_ijkm1
    S_oil_ijk     = 1-S_water_ijk
    S_oil_ijkp1   = 1-S_water_ijkp1
    S_oil_ijp1k   = 1-S_water_ijp1k
    S_oil_ip1jk   = 1-S_water_ip1jk
    S_oil_prev   = 1-S_water_prev

    # Pressure
    p_oil_im1jk   = g[1,1]
    p_oil_ijm1k   = g[2,1]
    p_oil_ijkm1   = g[3,1]
    p_oil_ijk     = g[4,1]
    p_oil_ijkp1   = g[5,1]
    p_oil_ijp1k   = g[6,1]
    p_oil_ip1jk   = g[7,1]
    p_oil_prev   = g_prev[i, j, k][1]

    p_water_im1jk = p_oil_im1jk - m.p_cow(S_water_im1jk)
    p_water_ijm1k = p_oil_ijm1k - m.p_cow(S_water_ijm1k)
    p_water_ijkm1 = p_oil_ijkm1 - m.p_cow(S_water_ijkm1)
    p_water_ijk   = p_oil_ijk   - m.p_cow(S_water_ijk)
    p_water_ijkp1 = p_oil_ijkp1 - m.p_cow(S_water_ijkp1)
    p_water_ijp1k = p_oil_ijp1k - m.p_cow(S_water_ijp1k)
    p_water_ip1jk = p_oil_ip1jk - m.p_cow(S_water_ip1jk)
    p_water_prev  = p_oil_prev  - m.p_cow(S_water_prev)


    # 5.615 is oil field units correction factor. See Note 4.
    V_ijk      = m.Δ[1][i,j,k]*m.Δ[2][i,j,k]*m.Δ[3][i,j,k]*m.ϕ[i,j,k]/5.615
    V_ijk_prev = m.Δ[1][i,j,k]*m.Δ[2][i,j,k]*m.Δ[3][i,j,k]*m.ϕ[i,j,k]/5.615

    # Fluid potentials
    Φ_water_im1jk = p_water_im1jk - m.ρ_w(p_water_im1jk)*m.z[im1,j,k]/144.0
    Φ_water_ijm1k = p_water_ijm1k - m.ρ_w(p_water_ijm1k)*m.z[i,jm1,k]/144.0
    Φ_water_ijkm1 = p_water_ijkm1 - m.ρ_w(p_water_ijkm1)*m.z[i,j,km1]/144.0
    Φ_water_ijk   = p_water_ijk   - m.ρ_w(p_water_ijk)  *m.z[i,j,k]  /144.0
    Φ_water_ijkp1 = p_water_ijkp1 - m.ρ_w(p_water_ijkp1)*m.z[i,j,kp1]/144.0
    Φ_water_ijp1k = p_water_ijp1k - m.ρ_w(p_water_ijp1k)*m.z[i,jp1,k]/144.0
    Φ_water_ip1jk = p_water_ip1jk - m.ρ_w(p_water_ip1jk)*m.z[ip1,j,k]/144.0

    Φ_oil_im1jk = p_oil_im1jk - m.ρ_o(p_oil_im1jk)*m.z[im1,j,k]/144.0
    Φ_oil_ijm1k = p_oil_ijm1k - m.ρ_o(p_oil_ijm1k)*m.z[i,jm1,k]/144.0
    Φ_oil_ijkm1 = p_oil_ijkm1 - m.ρ_o(p_oil_ijkm1)*m.z[i,j,km1]/144.0
    Φ_oil_ijk   = p_oil_ijk   - m.ρ_o(p_oil_ijk)  *m.z[i,j,k]  /144.0
    Φ_oil_ijkp1 = p_oil_ijkp1 - m.ρ_o(p_oil_ijkp1)*m.z[i,j,kp1]/144.0
    Φ_oil_ijp1k = p_oil_ijp1k - m.ρ_o(p_oil_ijp1k)*m.z[i,jp1,k]/144.0
    Φ_oil_ip1jk = p_oil_ip1jk - m.ρ_o(p_oil_ip1jk)*m.z[ip1,j,k]/144.0
    Φ_bth       = p_bth       - m.ρ_o(p_bth)*m.z[i,j,k]/144.0

    # Upstream condition
    k_r_water_west  = Φ_water_im1jk > Φ_water_ijk ? m.k_r_w(S_water_im1jk)*m.ρ_w(p_water_im1jk) : m.k_r_w(S_water_ijk)*m.ρ_w(p_water_ijk)
    k_r_water_east  = Φ_water_ip1jk > Φ_water_ijk ? m.k_r_w(S_water_ip1jk)*m.ρ_w(p_water_ip1jk) : m.k_r_w(S_water_ijk)*m.ρ_w(p_water_ijk)
    k_r_water_south = Φ_water_ijm1k > Φ_water_ijk ? m.k_r_w(S_water_ijm1k)*m.ρ_w(p_water_ijm1k) : m.k_r_w(S_water_ijk)*m.ρ_w(p_water_ijk)
    k_r_water_north = Φ_water_ijp1k > Φ_water_ijk ? m.k_r_w(S_water_ijp1k)*m.ρ_w(p_water_ijp1k) : m.k_r_w(S_water_ijk)*m.ρ_w(p_water_ijk)
    k_r_water_below = Φ_water_ijkm1 > Φ_water_ijk ? m.k_r_w(S_water_ijkm1)*m.ρ_w(p_water_ijkm1) : m.k_r_w(S_water_ijk)*m.ρ_w(p_water_ijk)
    k_r_water_above = Φ_water_ijkp1 > Φ_water_ijk ? m.k_r_w(S_water_ijkp1)*m.ρ_w(p_water_ijkp1) : m.k_r_w(S_water_ijk)*m.ρ_w(p_water_ijk)


    k_r_oil_west  = Φ_oil_im1jk > Φ_oil_ijk ? m.k_r_o(S_water_im1jk)*m.ρ_o(p_oil_im1jk) : m.k_r_o(S_water_ijk)*m.ρ_o(p_oil_ijk)
    k_r_oil_east  = Φ_oil_ip1jk > Φ_oil_ijk ? m.k_r_o(S_water_ip1jk)*m.ρ_o(p_oil_ip1jk) : m.k_r_o(S_water_ijk)*m.ρ_o(p_oil_ijk)
    k_r_oil_south = Φ_oil_ijm1k > Φ_oil_ijk ? m.k_r_o(S_water_ijm1k)*m.ρ_o(p_oil_ijm1k) : m.k_r_o(S_water_ijk)*m.ρ_o(p_oil_ijk)
    k_r_oil_north = Φ_oil_ijp1k > Φ_oil_ijk ? m.k_r_o(S_water_ijp1k)*m.ρ_o(p_oil_ijp1k) : m.k_r_o(S_water_ijk)*m.ρ_o(p_oil_ijk)
    k_r_oil_below = Φ_oil_ijkm1 > Φ_oil_ijk ? m.k_r_o(S_water_ijkm1)*m.ρ_o(p_oil_ijkm1) : m.k_r_o(S_water_ijk)*m.ρ_o(p_oil_ijk)
    k_r_oil_above = Φ_oil_ijkp1 > Φ_oil_ijk ? m.k_r_o(S_water_ijkp1)*m.ρ_o(p_oil_ijkp1) : m.k_r_o(S_water_ijk)*m.ρ_o(p_oil_ijk)


    Δx_west  = (m.Δ[1][im1,j,k] + m.Δ[1][i,j])/2
    Δx_east  = (m.Δ[1][ip1,j,k] + m.Δ[1][i,j])/2
    Δy_south = (m.Δ[2][i,jm1,k] + m.Δ[2][i,j])/2
    Δy_north = (m.Δ[2][i,jp1,k] + m.Δ[2][i,j])/2
    Δz_below = (m.Δ[3][i,j,km1] + m.Δ[3][i,j])/2
    Δz_above = (m.Δ[3][i,j,kp1] + m.Δ[3][i,j])/2


    # The 1.127e-3 factor is oil field units. See Note 4.

    T_water_west  = 1.127e-3*k_west*k_r_water_west/m.μ_w*A_west/Δx_west # boundary condition
    T_water_east  = 1.127e-3*k_east*k_r_water_east/m.μ_w*A_east/Δx_east
    T_water_south = 1.127e-3*k_south*k_r_water_south/m.μ_w*A_south/Δy_south
    T_water_north = 1.127e-3*k_north*k_r_water_north/m.μ_w*A_north/Δy_north
    T_water_below = 1.127e-3*k_below*k_r_water_below/m.μ_w*A_below/Δz_below
    T_water_above = 1.127e-3*k_above*k_r_water_above/m.μ_w*A_above/Δz_above

    T_oil_west    = 1.127e-3*k_west*k_r_oil_west/m.μ_o*A_west/Δx_west
    T_oil_east    = 1.127e-3*k_east*k_r_oil_east/m.μ_o*A_east/Δx_east
    T_oil_south   = 1.127e-3*k_south*k_r_oil_south/m.μ_o*A_south/Δy_south
    T_oil_north   = 1.127e-3*k_north*k_r_oil_north/m.μ_o*A_north/Δy_north
    T_oil_below   = 1.127e-3*k_below*k_r_oil_below/m.μ_o*A_below/Δz_below
    T_oil_above   = 1.127e-3*k_above*k_r_oil_above/m.μ_o*A_above/Δz_above

    PI = bth ? 7.06e-3*(mean([m.k[i,j,k][direc] for direc in 1:3]))*m.Δ[3][i,j,k]/m.μ_o/log(0.2*m.Δ[1][i,j,k]/0.416) : 0.0
    if q[2]==0.0 PI = 0 end

    residual_water_ijk = T_water_west*(Φ_water_im1jk - Φ_water_ijk) +
                       T_water_east*(Φ_water_ip1jk - Φ_water_ijk)   +
                       T_water_south*(Φ_water_ijm1k - Φ_water_ijk)  +
                       T_water_north*(Φ_water_ijp1k - Φ_water_ijk)  +
                       T_water_above*(Φ_water_ijkp1 - Φ_water_ijk)  +
                       T_water_below*(Φ_water_ijkm1 - Φ_water_ijk)  -
                       q_water*m.ρ_w(p_water_ijk)                   - # injection
                       (V_ijk*S_water_ijk*m.ρ_w(p_water_ijk)        -
                       V_ijk_prev*S_water_prev*m.ρ_w(p_water_prev))/m.Δt

    residual_oil_ijk   = T_oil_west*(Φ_oil_im1jk - Φ_oil_ijk)      +
                       T_oil_east*(Φ_oil_ip1jk - Φ_oil_ijk)        +
                       T_oil_south*(Φ_oil_ijm1k - Φ_oil_ijk)       +
                       T_oil_north*(Φ_oil_ijp1k - Φ_oil_ijk)       +
                       T_oil_above*(Φ_oil_ijkp1 - Φ_oil_ijk)       +
                       T_oil_below*(Φ_oil_ijkm1 - Φ_oil_ijk)       -
                       q_oil*m.ρ_o(p_oil_ijk)                      - # production
                       PI*(Φ_water_ijk-Φ_bth)                      -
                       (V_ijk*S_oil_ijk*m.ρ_o(p_oil_ijk)           -
                       V_ijk_prev*S_oil_prev*m.ρ_o(p_oil_prev))/m.Δt


    return [residual_water_ijk, residual_oil_ijk]
end

function res_each{T}(m, q, g::Grid{T,3,7,Array{T,3}}, g_prev::Grid{T,3,7,Array{T,3}}, i, j, k)
    return res_f(m, q[i,j,k,:], [g[i-1,j,k][1] g[i-1,j,k][2]; g[i,j-1,k][1] g[i,j-1,k][2];
                  g[i,j,k-1][1] g[i,j,k-1][2]; g[i,j,k][1] g[i,j,k][2];
                  g[i,j,k+1][1] g[i,j,k+1][2]; g[i,j+1,k][1] g[i,j+1,k][2];
                  g[i+1,j,k][1] g[i+1,j,k][2]], g_prev, i, j, k)
end
function getresidual{T}(m, q, g::Grid{T,3,7,Array{T,3}}, g_prev::Grid{T,3,7,Array{T,3}})
    Nx, Ny, Nz = size(m.Δ[1])
    return makegrid([SVector{2,Float64}(res_each(m, q, g, g_prev, i, j, k)) for i=1:Nx, j=1:Ny, k=1:Nz], 7)
end
function getlocalresidual{T}(m, q, g::Grid{T,3,7,Array{T,3}}, g_prev::Grid{T,3,7,Array{T,3}})
    Nx, Ny, Nz = size(m.Δ[1])
    return [SVector{2,Float64}(res_each(m, q, g, g_prev, i, j, k)) for i=1:Nx, j=1:Ny, k=1:Nz]
end
function getstencilArray{T}(m, q, g::Grid{T,3,7,Array{T,3}}, g_prev::Grid{T,3,7,Array{T,3}})
    Nx, Ny, Nz = size(m.Δ[1])
    stencilArray = Array{StencilPoint{SMatrix{2,2,Float64,4},3,7},3}(Nx, Ny, Nz)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        J = ForwardDiff.jacobian(θ -> res_f(m, q[i,j,k,:], θ, g_prev, i, j, k), [g[i-1,j,k][1] g[i-1,j,k][2]; g[i,j-1,k][1] g[i,j-1,k][2];
                                                      g[i,j,k-1][1] g[i,j,k-1][2]; g[i,j,k][1] g[i,j,k][2];
                                                      g[i,j,k+1][1] g[i,j,k+1][2]; g[i,j+1,k][1] g[i,j+1,k][2]; g[i+1,j,k][1] g[i+1,j,k][2]])
        stencilArray[i,j,k] = StencilPoint{SMatrix{2,2,Float64,4},3,7}(
            (@SMatrix([J[1,1] J[1,8]; J[2,1] J[2,8]]),
             @SMatrix([J[1,2] J[1,9]; J[2,2] J[2,9]]),
             @SMatrix([J[1,3] J[1,10]; J[2,3] J[2,10]]),
             @SMatrix([J[1,4] J[1,11]; J[2,4] J[2,11]]),
             @SMatrix([J[1,5] J[1,12]; J[2,5] J[2,12]]),
             @SMatrix([J[1,6] J[1,13]; J[2,6] J[2,13]]),
             @SMatrix([J[1,7] J[1,14]; J[2,7] J[2,14]]),))
    end
    return stencilArray
end
