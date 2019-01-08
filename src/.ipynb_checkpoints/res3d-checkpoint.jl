using ForwardDiff

function res_f(m, Δt, g, g_prev, i, j, k;bth=true, p_bth=4000.0, maxinj = 10000.0)

    Nx, Ny, Nz = size(m)

    # Injection
    q_w = g[4,1]>maxinj ? 0.0 : m.q_water[i,j,k]  ## if pressure is larger than max, no injection

    im1 = max(1, i-1)
    ip1 = min(Nx, i+1)
    jm1 = max(1, j-1)
    jp1 = min(Ny, j+1)
    km1 = max(1, k-1)
    kp1 = min(Nz, k+1)

    ###=====================================================================
    ### Compute inter-block quantities
    ###=====================================================================   
    Δx_west  = (m.Δ[1][im1,j,k] + m.Δ[1][i,j,k])/2
    Δx_east  = (m.Δ[1][ip1,j,k] + m.Δ[1][i,j,k])/2
    Δy_south = (m.Δ[2][i,jm1,k] + m.Δ[2][i,j,k])/2
    Δy_north = (m.Δ[2][i,jp1,k] + m.Δ[2][i,j,k])/2
    Δz_below = (m.Δ[3][i,j,km1] + m.Δ[3][i,j,k])/2
    Δz_above = (m.Δ[3][i,j,kp1] + m.Δ[3][i,j,k])/2
    
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
    

    ###=====================================================================
    ### Load S, p and use Aux equation
    ###=====================================================================   
    #Saturation load
    S_w_im1jk   = g[1,2]
    S_w_ijm1k   = g[2,2]
    S_w_ijkm1   = g[3,2]
    S_w_ijk     = g[4,2]
    S_w_ijkp1   = g[5,2]
    S_w_ijp1k   = g[6,2]
    S_w_ip1jk   = g[7,2]
    S_w_prev    = g_prev[i,j,k,2]

    # Using Aux Eq
    S_o_im1jk   = 1-S_w_im1jk
    S_o_ijm1k   = 1-S_w_ijm1k
    S_o_ijkm1   = 1-S_w_ijkm1
    S_o_ijk     = 1-S_w_ijk
    S_o_ijkp1   = 1-S_w_ijkp1
    S_o_ijp1k   = 1-S_w_ijp1k
    S_o_ip1jk   = 1-S_w_ip1jk
    S_o_prev    = 1-S_w_prev

    # Pressure load
    p_o_im1jk   = g[1,1]
    p_o_ijm1k   = g[2,1]
    p_o_ijkm1   = g[3,1]
    p_o_ijk     = g[4,1]
    p_o_ijkp1   = g[5,1]
    p_o_ijp1k   = g[6,1]
    p_o_ip1jk   = g[7,1]
    p_o_prev   = g_prev[i,j,k,1]

    # Capillary pressure
    p_w_im1jk = p_o_im1jk - m.p_cow(S_w_im1jk)
    p_w_ijm1k = p_o_ijm1k - m.p_cow(S_w_ijm1k)
    p_w_ijkm1 = p_o_ijkm1 - m.p_cow(S_w_ijkm1)
    p_w_ijk   = p_o_ijk   - m.p_cow(S_w_ijk)
    p_w_ijkp1 = p_o_ijkp1 - m.p_cow(S_w_ijkp1)
    p_w_ijp1k = p_o_ijp1k - m.p_cow(S_w_ijp1k)
    p_w_ip1jk = p_o_ip1jk - m.p_cow(S_w_ip1jk)
    p_w_prev  = p_o_prev  - m.p_cow(S_w_prev)

    # 5.615 is oil field units correction factor. See Note 3.
    V_ijk      = m.Δ[1][i,j,k]*m.Δ[2][i,j,k]*m.Δ[3][i,j,k]*m.ϕ[i,j,k]/5.615
    V_ijk_prev = m.Δ[1][i,j,k]*m.Δ[2][i,j,k]*m.Δ[3][i,j,k]*m.ϕ[i,j,k]/5.615
    
    
    ###=====================================================================
    ### Calculate Fluid potentials Φ
    ###=====================================================================
    Φ_w_im1jk = p_w_im1jk - m.ρ_w(p_w_im1jk)*m.z[im1,j,k]/144.0
    Φ_w_ijm1k = p_w_ijm1k - m.ρ_w(p_w_ijm1k)*m.z[i,jm1,k]/144.0
    Φ_w_ijkm1 = p_w_ijkm1 - m.ρ_w(p_w_ijkm1)*m.z[i,j,km1]/144.0
    Φ_w_ijk   = p_w_ijk   - m.ρ_w(p_w_ijk)  *m.z[i,j,k]  /144.0
    Φ_w_ijkp1 = p_w_ijkp1 - m.ρ_w(p_w_ijkp1)*m.z[i,j,kp1]/144.0
    Φ_w_ijp1k = p_w_ijp1k - m.ρ_w(p_w_ijp1k)*m.z[i,jp1,k]/144.0
    Φ_w_ip1jk = p_w_ip1jk - m.ρ_w(p_w_ip1jk)*m.z[ip1,j,k]/144.0

    Φ_o_im1jk = p_o_im1jk - m.ρ_o(p_o_im1jk)*m.z[im1,j,k]/144.0
    Φ_o_ijm1k = p_o_ijm1k - m.ρ_o(p_o_ijm1k)*m.z[i,jm1,k]/144.0
    Φ_o_ijkm1 = p_o_ijkm1 - m.ρ_o(p_o_ijkm1)*m.z[i,j,km1]/144.0
    Φ_o_ijk   = p_o_ijk   - m.ρ_o(p_o_ijk)  *m.z[i,j,k]  /144.0
    Φ_o_ijkp1 = p_o_ijkp1 - m.ρ_o(p_o_ijkp1)*m.z[i,j,kp1]/144.0
    Φ_o_ijp1k = p_o_ijp1k - m.ρ_o(p_o_ijp1k)*m.z[i,jp1,k]/144.0
    Φ_o_ip1jk = p_o_ip1jk - m.ρ_o(p_o_ip1jk)*m.z[ip1,j,k]/144.0

    
    ###=====================================================================
    ### Compute Relative Permeability
    ###=====================================================================
    # Upstream condition. Relative permeability is always a function of S_water!    
    k_r_w_west  = Φ_w_im1jk > Φ_w_ijk ? m.k_r_w(S_w_im1jk)*m.ρ_w(p_w_im1jk) : m.k_r_w(S_w_ijk)*m.ρ_w(p_w_ijk)
    k_r_w_east  = Φ_w_ip1jk > Φ_w_ijk ? m.k_r_w(S_w_ip1jk)*m.ρ_w(p_w_ip1jk) : m.k_r_w(S_w_ijk)*m.ρ_w(p_w_ijk)
    k_r_w_south = Φ_w_ijm1k > Φ_w_ijk ? m.k_r_w(S_w_ijm1k)*m.ρ_w(p_w_ijm1k) : m.k_r_w(S_w_ijk)*m.ρ_w(p_w_ijk)
    k_r_w_north = Φ_w_ijp1k > Φ_w_ijk ? m.k_r_w(S_w_ijp1k)*m.ρ_w(p_w_ijp1k) : m.k_r_w(S_w_ijk)*m.ρ_w(p_w_ijk)
    k_r_w_below = Φ_w_ijkm1 > Φ_w_ijk ? m.k_r_w(S_w_ijkm1)*m.ρ_w(p_w_ijkm1) : m.k_r_w(S_w_ijk)*m.ρ_w(p_w_ijk)
    k_r_w_above = Φ_w_ijkp1 > Φ_w_ijk ? m.k_r_w(S_w_ijkp1)*m.ρ_w(p_w_ijkp1) : m.k_r_w(S_w_ijk)*m.ρ_w(p_w_ijk)


    k_r_o_west  = Φ_o_im1jk > Φ_o_ijk ? m.k_r_o(S_w_im1jk)*m.ρ_o(p_o_im1jk) : m.k_r_o(S_w_ijk)*m.ρ_o(p_o_ijk)
    k_r_o_east  = Φ_o_ip1jk > Φ_o_ijk ? m.k_r_o(S_w_ip1jk)*m.ρ_o(p_o_ip1jk) : m.k_r_o(S_w_ijk)*m.ρ_o(p_o_ijk)
    k_r_o_south = Φ_o_ijm1k > Φ_o_ijk ? m.k_r_o(S_w_ijm1k)*m.ρ_o(p_o_ijm1k) : m.k_r_o(S_w_ijk)*m.ρ_o(p_o_ijk)
    k_r_o_north = Φ_o_ijp1k > Φ_o_ijk ? m.k_r_o(S_w_ijp1k)*m.ρ_o(p_o_ijp1k) : m.k_r_o(S_w_ijk)*m.ρ_o(p_o_ijk)
    k_r_o_below = Φ_o_ijkm1 > Φ_o_ijk ? m.k_r_o(S_w_ijkm1)*m.ρ_o(p_o_ijkm1) : m.k_r_o(S_w_ijk)*m.ρ_o(p_o_ijk)
    k_r_o_above = Φ_o_ijkp1 > Φ_o_ijk ? m.k_r_o(S_w_ijkp1)*m.ρ_o(p_o_ijkp1) : m.k_r_o(S_w_ijk)*m.ρ_o(p_o_ijk)
    

    ###=====================================================================
    ### Calculate Interblock Transmissibility
    ###=====================================================================
    # The 1.127e-3 factor is oil field units. See Note 4.
    T_w_west  = 1.127e-3*k_west*k_r_w_west/m.μ_w*A_west/Δx_west # boundary condition
    T_w_east  = 1.127e-3*k_east*k_r_w_east/m.μ_w*A_east/Δx_east
    T_w_south = 1.127e-3*k_south*k_r_w_south/m.μ_w*A_south/Δy_south
    T_w_north = 1.127e-3*k_north*k_r_w_north/m.μ_w*A_north/Δy_north
    T_w_below = 1.127e-3*k_below*k_r_w_below/m.μ_w*A_below/Δz_below
    T_w_above = 1.127e-3*k_above*k_r_w_above/m.μ_w*A_above/Δz_above

    T_o_west    = 1.127e-3*k_west*k_r_o_west/m.μ_o*A_west/Δx_west
    T_o_east    = 1.127e-3*k_east*k_r_o_east/m.μ_o*A_east/Δx_east
    T_o_south   = 1.127e-3*k_south*k_r_o_south/m.μ_o*A_south/Δy_south
    T_o_north   = 1.127e-3*k_north*k_r_o_north/m.μ_o*A_north/Δy_north
    T_o_below   = 1.127e-3*k_below*k_r_o_below/m.μ_o*A_below/Δz_below
    T_o_above   = 1.127e-3*k_above*k_r_o_above/m.μ_o*A_above/Δz_above

    
    ###=====================================================================
    ### Impose Well Condition
    ###=====================================================================
    # Calculate Wellbore Radius(SPE-10 Config), Peaceman Radius r_e (Chen. 449p)
    k12, k21 = (m.k[i,j,k][1]/m.k[i,j,k][2]), (m.k[i,j,k][2]/m.k[i,j,k][1])
    r_w      = 5.0/12.0 # SPE10 Config
    r_e      = (sqrt(2)/5)*sqrt(sqrt(k21)*m.Δ[1][i,j,k]^2+sqrt(k12)*m.Δ[2][i,j,k]^2)/(k12^0.25+k21^0.25)
     
    # Calculate Productivity index (Well-Index) if it is production well
    PI = m.q_oil[i,j,k] > 0 ? 7.06e-3*((m.k[i,j,k][1]))*m.Δ[3][i,j,k]/log(r_e/r_w) : 0.0
    
    # 120001 is depth of center of the surface level grids
    Φ_diff_o   = p_o_ijk-p_bth-m.ρ_o(p_o_ijk)*(m.z[i,j,k]-12001)/144
    Φ_diff_w   = p_w_ijk-p_bth-m.ρ_w(p_w_ijk)*(m.z[i,j,k]-12001)/144
    well_o     = PI==0.0 ? 0.0 : PI*Φ_diff_o*m.k_r_o(S_w_ijk)*m.ρ_o(p_o_ijk)/m.μ_o
    well_w     = PI==0.0 ? 0.0 : PI*Φ_diff_w*m.k_r_w(S_w_ijk)*m.ρ_w(p_w_ijk)/m.μ_w

    
    ###=====================================================================
    ### Calculate Residuals
    ###=====================================================================
    residual_water_ijk = T_w_west*(Φ_w_im1jk - Φ_w_ijk)   +
                         T_w_east*(Φ_w_ip1jk - Φ_w_ijk)   +
                         T_w_south*(Φ_w_ijm1k - Φ_w_ijk)  +
                         T_w_north*(Φ_w_ijp1k - Φ_w_ijk)  +
                         T_w_above*(Φ_w_ijkp1 - Φ_w_ijk)  +
                         T_w_below*(Φ_w_ijkm1 - Φ_w_ijk)  -
                         q_w*m.ρ_w(p_w_ijk)                -
                         well_w                            -
                         (V_ijk*S_w_ijk*m.ρ_w(p_w_ijk)     -
                         V_ijk_prev*S_w_prev*m.ρ_w(p_w_prev))/Δt
    
    residual_oil_ijk   = T_o_west*(Φ_o_im1jk - Φ_o_ijk)   +
                         T_o_east*(Φ_o_ip1jk - Φ_o_ijk)   +
                         T_o_south*(Φ_o_ijm1k - Φ_o_ijk)  +
                         T_o_north*(Φ_o_ijp1k - Φ_o_ijk)  +
                         T_o_above*(Φ_o_ijkp1 - Φ_o_ijk)  +
                         T_o_below*(Φ_o_ijkm1 - Φ_o_ijk)  -
                         well_o                            -
                         (V_ijk*S_o_ijk*m.ρ_o(p_o_ijk)     -
                         V_ijk_prev*S_o_prev*m.ρ_o(p_o_prev))/Δt
    
    
    return [residual_water_ijk, residual_oil_ijk]
end

function res_each(m, Δt, g::MGrid{2,T,3,7,Array{T,3}}, g_prev::MGrid{2,T,3,7,Array{T,3}}, i, j, k) where {T}
    return res_f(m, Δt, [g[i-1,j,k,1] g[i-1,j,k,2]; g[i,j-1,k,1] g[i,j-1,k,2];
                                    g[i,j,k-1,1] g[i,j,k-1,2]; g[i,j,k,1] g[i,j,k,2];
                                    g[i,j,k+1,1] g[i,j,k+1,2]; g[i,j+1,k,1] g[i,j+1,k,2];
                                    g[i+1,j,k,1] g[i+1,j,k,2]], g_prev, i, j, k)
end
function getresidual(m, Δt, g::MGrid{2,T,3,7,Array{T,3}}, g_prev::MGrid{2,T,3,7,Array{T,3}}) where {T}## Need modification. Inefficient derivation
    Nx, Ny, Nz = size(m)
    tmp = [res_each(m, Δt, g, g_prev, i, j, k) for i=1:Nx, j=1:Ny, k=1:Nz]
    return MGrid{2,T,3,7,Array{T,3}}((makegrid([tmp[i,j,k][1] for i in 1:Nx, j in 1:Ny, k in 1:Nz],7),makegrid([tmp[i,j,k][2] for i in 1:Nx, j in 1:Ny, k in 1:Nz],7)))
end
function getlocalresidual(m, Δt, g::MGrid{2,T,3,7,Array{T,3}}, g_prev::MGrid{2,T,3,7,Array{T,3}}) where {T}## Need modification. Inefficient derivation
    Nx, Ny, Nz = size(m)
    return [res_each(m, Δt, g, g_prev, i, j, k) for i=1:Nx, j=1:Ny, k=1:Nz]
end
function getstencilArray(m, Δt, g::MGrid{2,T,3,7,Array{T,3}}, g_prev::MGrid{2,T,3,7,Array{T,3}}) where {T}
    Nx, Ny, Nz = size(m)
    stencilArray1 = Array{StencilPoint{Float64,3,7},3}(undef,Nx, Ny, Nz)
    stencilArray2 = Array{StencilPoint{Float64,3,7},3}(undef,Nx, Ny, Nz)
    stencilArray3 = Array{StencilPoint{Float64,3,7},3}(undef,Nx, Ny, Nz)
    stencilArray4 = Array{StencilPoint{Float64,3,7},3}(undef,Nx, Ny, Nz)
    for i in 1:Nx, j in 1:Ny, k in 1:Nz
        J = ForwardDiff.jacobian(θ -> res_f(m, Δt, θ, g_prev, i, j, k), [g[i-1,j,k,1] g[i-1,j,k,2]; g[i,j-1,k,1] g[i,j-1,k,2];
                                                                                    g[i,j,k-1,1] g[i,j,k-1,2]; g[i,j,k,1] g[i,j,k,2];
                                                                                    g[i,j,k+1,1] g[i,j,k+1,2]; g[i,j+1,k,1] g[i,j+1,k,2]; g[i+1,j,k,1] g[i+1,j,k,2]])
        stencilArray1[i,j,k] = StencilPoint{Float64,3,7}((J[1,1],J[1,2],J[1,3],J[1,4],J[1,5],J[1,6],J[1,7]))
        stencilArray2[i,j,k] = StencilPoint{Float64,3,7}((J[1,8],J[1,9],J[1,10],J[1,11],J[1,12],J[1,13],J[1,14]))
        stencilArray3[i,j,k] = StencilPoint{Float64,3,7}((J[2,1],J[2,2],J[2,3],J[2,4],J[2,5],J[2,6],J[2,7]))
        stencilArray4[i,j,k] = StencilPoint{Float64,3,7}((J[2,8],J[2,9],J[2,10],J[2,11],J[2,12],J[2,13],J[2,14]))
    end
    return stencilArray1, stencilArray2, stencilArray3, stencilArray4
end
