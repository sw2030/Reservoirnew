using CuArrays, Distributed, DistributedArrays, LinearAlgebra

function CutoD(custencil::GPUStencil{T,N,P}) where{T,N,P}
    dA = distribute(Array(custencil.v));
    return Stencil{T,N,P,typeof(dA)}(dA)
end
function DtoCu(dstencil::DStencil{T,N,P}) where {T,N,P}
    cA = CuArray(Array(dstencil.v))
    return Stencil{T,N,P,typeof(cA)}(cA)
end
DtoCu(dmstencil::DMStencil) = MStencil(DtoCu.(dmstencil.stencils))
CutoD(cmstencil::GPUMStencil) = MStencil(CutoD.(cmstencil.stencils))


function CutoD(cugrid::Grid{T,N,P,<:CuArray}) where {T,N,P}
    cua = Array(cugrid.A) ## Make it into normal array
    tmp = distribute(@view cua[2:end-1, 2:end-1, 2:end-1])
    dA  = makegrid(tmp, P)
    close(tmp)
    return dA
end
function DtoCu(dgrid::DGrid{T,3,P}) where {T,P}
    da = zeros(T, size(dgrid,1)+2, size(dgrid,2)+2, size(dgrid,3)+2)
    da[2:end-1, 2:end-1, 2:end-1] = Array(dgrid.A)
    return Grid{T,3,P,CuArray{T,3}}(CuArray(da))
end
CutoD(cugrid::GPUMGrid) = CutoD.(cugrid)
DtoCu(dgrid::DMGrid)    = DtoCu.(dgrid)

###=====================================================================
### Solver
###=====================================================================
function SPE10Solve(m::Reservoirmodel, Δt_plan, g_guess; tol_relnorm=1e-2, tol_gmres=1e-2, n_restart=20, n_iter=1000, precondf=Reservoir.precond_1)
    
    ## Check Backend
    if typeof(g_guess[1].A) <: CuArray
        println("Backend : GPU")
        backend = 'G'
    elseif typeof(g_guess[1].A) <: DArray
        println("Backend : Distributed CPU")
        backend = 'D'
    else
        println("No parallel backend detected")
        backend = 'A'
    end    
    
    
    ## Initialize
    record_p   = zeros(2, length(Δt_plan)-1)
    psgrid_old = copy(g_guess)
    psgrid_new = copy(psgrid_old)
    
    
    ## Time stepping start
    for steps in 1:length(Δt_plan)-1
        Δt = Δt_plan[steps+1]
        
        RES = getresidual(m, Δt, psgrid_new, psgrid_old)
        norm_RES_save = norm(RES)
        norm_RES = norm_RES_save
        println("\nstep ", steps, " | norm_RES : ", norm_RES, " | Δt : ",Δt, " | ΣΔt : ", sum(Δt_plan[1:steps+1]))
        gmresnumcount, gmresitercount = 0, 0
        while(norm_RES/norm_RES_save > tol_relnorm)
            
            ## In case it is diverging
            if (norm_RES > 1.0e7 || gmresnumcount > 9)
                copyto!(psgrid_new, psgrid_old)
                Δt *= 0.5
                Δt_plan[steps+1] *= 0.5
                println("New Δt adapted... | Δt : ",Δt, " | ΣΔt : ", sum(Δt_plan[1:steps+1]))
                gmresnumcount, gmresitercount = 0, 0
                if backend=='D' close(RES) end
                RES = getresidual(m, Δt, psgrid_new, psgrid_old)
            end
            
            JAC     = getstencil(m, Δt, psgrid_new, psgrid_old)
            JAC_GPU = DtoCu(JAC)
            RES_GPU = DtoCu(RES)
            precP, precE = Reservoir.make_P_E_precond_1(JAC_GPU)
            print("GMRES start...")
            gmresresult = stencilgmres2(JAC_GPU, RES_GPU, n_restart; tol=tol_gmres, maxiter=n_iter, M=(t->precondf(precP,precE,t)), ifprint=false)
            gmresitercount += gmresresult[3]
            gmresnumcount  += 1
            if backend=='D' close(RES), close(JAC) end
            println("...GMRES done  ||  Iter : ", gmresresult[3])
            gmres_dist = CutoD(gmresresult[1])
            LinearAlgebra.axpy!(-1.0, gmres_dist, psgrid_new)
            RES = getresidual(m, Δt, psgrid_new, psgrid_old)
            norm_RES, norm_dg = norm(RES), norm(gmresresult[1])
            if backend=='D' close(gmres_dist) end
            @show norm_RES, norm_dg        
        end
        if backend=='D' close(RES) end
        copyto!(psgrid_old, psgrid_new)
        record_p[:, steps] = [sum(Δt_plan[1:steps+1]); sum(Array(psgrid_old[1].A))/60/220/85]
        println("Total GMRES iteration : ",gmresitercount, " | Avg p : ", record_p[2, steps])
    end
    if backend=='D' close(psgrid_old) end
    print("\nSolve done")
    return psgrid_new, record_p
end
function SPE10adaptivesolve(m::Reservoirmodel, t_init, Δt, g_guess, n_steps; tol_relnorm=1e-2, tol_gmres=1e-2, n_restart=20, n_iter=1000, precondf=Reservoir.precond_1)
    
    ## Check Backend
    if typeof(g_guess[1].A) <: CuArray
        println("Backend : GPU")
        backend = 'G'
    elseif typeof(g_guess[1].A) <: DArray
        println("Backend : Distributed CPU")
        backend = 'D'
    else
        println("No parallel backend detected")
        backend = 'A'
    end    
    
    
    ## Initialize
    record_p   = zeros(2, n_steps)
    psgrid_old = copy(g_guess)
    psgrid_new = copy(psgrid_old)
    
    
    ## Time stepping start
    for steps in 1:n_steps
        RES = getresidual(m, Δt, psgrid_new, psgrid_old)
        norm_RES_save = norm(RES)
        norm_RES = norm_RES_save
        println("\nstep ", steps, " | norm_RES : ", norm_RES, " | Δt : ",Δt, " | ΣΔt : ", t_init+Δt)
        gmresnumcount, gmresitercount = 0, 0
        while(norm_RES/norm_RES_save > tol_relnorm)
            
            ## In case it is diverging
            if (norm_RES > 1.0e7 || gmresnumcount > 9)
                copyto!(psgrid_new, psgrid_old)
                Δt *= 0.5
                println("\nNew Δt adapted... | Δt : ",Δt, " | ΣΔt : ", t_init+Δt)
                gmresnumcount, gmresitercount = 0, 0
                if backend=='D' close(RES) end
                RES = getresidual(m, Δt, psgrid_new, psgrid_old)
            end
            
            JAC     = getstencil(m, Δt, psgrid_new, psgrid_old)
            JAC_GPU = DtoCu(JAC)
            RES_GPU = DtoCu(RES)
            precP, precE = Reservoir.make_P_E_precond_1(JAC_GPU)
            print("GMRES start...")
            gmresresult = stencilgmres2(JAC_GPU, RES_GPU, n_restart; tol=tol_gmres, maxiter=n_iter, M=(t->precondf(precP,precE,t)), ifprint=false)
            gmresitercount += gmresresult[3]
            gmresnumcount  += 1
            if backend=='D' close(RES), close(JAC) end
            println("...GMRES done  ||  Iter : ", gmresresult[3])
            gmres_dist = CutoD(gmresresult[1])
            LinearAlgebra.axpy!(-1.0, gmres_dist, psgrid_new)
            RES = getresidual(m, Δt, psgrid_new, psgrid_old)
            norm_RES, norm_dg = norm(RES), norm(gmresresult[1])
            if backend=='D' close(gmres_dist) end
            @show norm_RES, norm_dg        
        end
        if backend=='D' close(RES) end
        copyto!(psgrid_old, psgrid_new)
        record_p[:, steps] = [t_init+Δt; sum(Array(psgrid_old[1].A))/60/220/85]
        println("Total GMRES iteration : ",gmresitercount, " | Avg p : ", record_p[2, steps])
        t_init += Δt
        Δt *= 2.0
    end
    if backend=='D' close(psgrid_old) end
    print("\nSolve done")
    return psgrid_new, record_p
end
    

















function ReservoirSolve(m::Reservoirmodel, Δt_plan, g_guess; tol_relnorm=1e-2, tol_gmres=1e-2, n_restart=20, n_iter=500, precondf=Reservoir.precond_1)
    
    ## Check Backend
    if typeof(g_guess[1].A) <: CuArray
        println("Backend : GPU")
        backend = 'G'
    elseif typeof(g_guess[1].A) <: DArray
        println("Backend : Distributed CPU")
        backend = 'D'
    else
        println("No parallel backend detected")
        backend = 'A'
    end    
    
    
    ## Initialize
    record_p   = zeros(2, length(Δt_plan)-1)
    psgrid_old = copy(g_guess)
    psgrid_new = copy(psgrid_old)
    
    
    ## Time stepping start
    for steps in 1:length(Δt_plan)-1
        Δt = Δt_plan[steps+1]
        
        RES = getresidual(m, Δt, psgrid_new, psgrid_old)
        norm_RES_save = norm(RES)
        norm_RES = norm_RES_save
        println("\nstep ", steps, " | norm_RES : ", norm_RES, " | Δt : ",Δt, " | ΣΔt : ", sum(Δt_plan[1:steps+1]))
        gmresnumcount, gmresitercount = 0, 0
        while(norm_RES/norm_RES_save > tol_relnorm)
            
            ## In case it is diverging
            if (norm_RES > 1.0e7 || gmresnumcount > 9)
                copyto!(psgrid_new, psgrid_old)
                Δt *= 0.5
                Δt_plan[steps+1] *= 0.5
                println("New Δt adapted... | Δt : ",Δt, " | ΣΔt : ", sum(Δt_plan[1:steps+1]))
                gmresnumcount, gmresitercount = 0, 0
                if backend=='D' close(RES) end
                RES = getresidual(m, Δt, psgrid_new, psgrid_old)
            end
            
            JAC = getstencil(m, Δt, psgrid_new, psgrid_old)
            precP, precE = Reservoir.make_P_E_precond_1(JAC)
            print("GMRES start...")
            gmresresult = stencilgmres2(JAC, RES, n_restart; tol=tol_gmres, maxiter=n_iter, M=(t->precondf(precP,precE,t)), ifprint=false)
            gmresitercount += gmresresult[3]
            gmresnumcount  += 1
            if backend=='D' close(precP), close(precE), close(RES), close(JAC) end
            println("...GMRES done  ||  Iter : ", gmresresult[3])
            LinearAlgebra.axpy!(-1.0, gmresresult[1], psgrid_new)
            RES = getresidual(m, Δt, psgrid_new, psgrid_old)
            norm_RES, norm_dg = norm(RES), norm(gmresresult[1])
            if backend=='D' close(gmresresult[1]) end
            @show norm_RES, norm_dg        
        end
        if backend=='D' close(RES) end
        copyto!(psgrid_old, psgrid_new)
        record_p[:, steps] = [sum(Δt_plan[1:steps+1]); sum(Array(psgrid_old[1].A))/60/220/85]
        println("Total GMRES iteration : ",gmresitercount, " | Avg p : ", record_p[2, steps])
    end
    if backend=='D' close(psgrid_old) end
    print("\nSolve done")
    return psgrid_new, record_p
end


function stencilgmres(A, b, restrt::Int64; tol::Real=1e-5, maxiter::Int=200, ifprint=false, M=identity, x = zero(b))
    realn, bnrm2 = gridsize(b), norm(b)
    if bnrm2==0 bnrm2 = 1.0 end
    r = copy(b)
    gemv!(-1, A, x, 1, r)
    if M!=identity copyto!(r, M(r)) end
    err = norm(r)/bnrm2
    itersave = 0
    ismax = false
    errlog = Float64[]

    if err<tol return x, ismax, itersave, err, errlog end

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
        if ifprint print(iter) end
        r = Q[1]
        copyto!(r, b)
        gemv!(-1, A, x, 1, r)
        if M!=identity copyto!(r, M(r)) end
        fill!(s, 0.0)
        s[1] = norm(r)
        rmul!(r, inv(s[1]))

        for i in 1:restrt
            isave = i
            w = Q[i+1]
            gemv!(1, A, Q[i], 0, w)
            if M!=identity copyto!(w, M(w)) end
            
            for k in 1:i
                H[k,i] = dot(w, Q[k])
                LinearAlgebra.axpy!(-H[k,i],Q[k],w)
            end
            H[i+1,i] = norm(w)
            rmul!(w, inv(H[i+1,i]))
            #Q[i+1] = w/H[i+1, i]
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
        ldiv!(UpperTriangular(view(H, 1:restrt, 1:restrt)), view(y, 1:restrt))  #x += Q[:,1:restrt]*y
        for k in 1:restrt
            LinearAlgebra.axpy!(y[k],Q[k],x)  #x += y[k]*Q[k]
        end
        copyto!(r, b)
        gemv!(-1, A, x, 1, r)
        copyto!(r, M(r))
        s[isave+1] = norm(r)
        err = s[isave+1]/bnrm2
        if err<=tol
            flag = 0
            break
        end
    end
    if flag==-1
        print("Maxiter")
        ismax = true
    end

    return x, ismax, itersave, err, errlog
end














function stencilgmres2(A::MStencil, b::MGrid, restrt::Int64; tol::Real=1e-5, maxiter::Int=200, ifprint=false, M=identity, x = zero(b))
    realn, bnrm2 = gridsize(b), norm(b)
    if bnrm2==0 bnrm2 = 1.0 end
    r = copy(b)
    gemv!(-1, A, M(x), 1, r)
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
    itersave = innerloop1!(Q, H, cs, sn, s, A, b, r, M, errlog, err, itersave, bnrm2, maxiter, isave, tol, y, x, flag, restrt)
    if flag==-1
        ifprint==true && print(" Maxiter")
        ismax = true
    end
    copyto!(x, M(x))
    return x, ismax, itersave, err, errlog
end
function innerloop1!(Q, H::Array{T,2}, cs::Array{T,1}, sn::Array{T,1}, s::Array{T,1}, A::MStencil, b::MGrid, r::MGrid, M::Function, errlog::Array{T,1}, err::T, itersave::Int64, bnrm2::T, maxiter::Int64, isave::Int64, tol::T, y::Array{T,1}, x::MGrid, flag::Int64, restrt::Int64) where {T}
    @inbounds for iter in 1:maxiter
        push!(errlog, err)
        itersave = iter
        r = Q[1]
        copyto!(r, b)    
        gemv!(-1, A, M(x), 1, r)   
        fill!(s, 0.0)
        tmp = norm(r)
        s[1] = norm(r) 
        rmul!(r, inv(s[1]))
        isave, err = innerloop2!(Q, H, cs, sn, s, A, M, restrt, isave, err, tol, bnrm2, y, x, flag)
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
        gemv!(-1, A, M(x), 1, r)
        s[isave+1] = norm(r)
        err = s[isave+1]/bnrm2
        if err<=tol
            flag = 0
            break
        end
    end
    return itersave
end
@inline function innerloop2!(Q, H::Array{T,2}, cs::Array{T,1}, sn::Array{T,1}, s::Array{T,1}, A::MStencil, M::Function, restrt::Int64, isave::Int64, err::T, tol::T, bnrm2::T, y::Array{T,1}, x::MGrid, flag::Int64) where {T}
    @inbounds for i in 1:restrt
        isave = i
        w = Q[i+1]
        gemv!(1, A, M(Q[i]), 0, w)  
        innerloop3!(Q, H, w, i)
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
    return isave, err
end
@inline function innerloop3!(Q, H::Array{T,2}, w::MGrid, i) where {T}
    @inbounds for k in 1:i
        H[k,i] = LinearAlgebra.dot(w, Q[k])
        LinearAlgebra.axpy!(-H[k,i],Q[k],w)
    end
end
#=
function innerloop1!(Q, H::Array{T,2}, cs::Array{T,1}, sn::Array{T,1}, s::Array{T,1}, A::MStencil, b::MGrid, r::MGrid, M::Function, errlog::Array{T,1}, err::T, itersave::Int64, bnrm2::T, maxiter::Int64, isave::Int64, tol::T, y::Array{T,1}, x::MGrid, flag::Int64, restrt::Int64) where {T}
    @inbounds for iter in 1:maxiter
        push!(errlog, err)
        itersave = iter
        r = Q[1]
        copyto!(r, b)    
        gemv!(-1, A, M(x), 1, r)   
        fill!(s, 0.0)
        tmp = norm(r)
        s[1] = norm(r) 
        rmul!(r, inv(s[1]))
        isave, err = innerloop2!(Q, H, cs, sn, s, A, M, restrt, isave, err, tol, bnrm2, y, x, flag)
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
        gemv!(-1, A, M(x), 1, r)
        s[isave+1] = norm(r)
        err = s[isave+1]/bnrm2
        if err<=tol
            flag = 0
            break
        end
    end
    return itersave
end
function innerloop2!(Q, H::Array{T,2}, cs::Array{T,1}, sn::Array{T,1}, s::Array{T,1}, A::MStencil, M::Function, restrt::Int64, isave::Int64, err::T, tol::T, bnrm2::T, y::Array{T,1}, x::MGrid, flag::Int64) where {T}
    @inbounds for i in 1:restrt
        isave = i
        w = Q[i+1]
        gemv!(1, A, M(Q[i]), 0, w)  
        innerloop3!(Q, H, w, i)
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
    return isave, err
end
function innerloop3!(Q, H::Array{T,2}, w::MGrid, i) where {T}
    @inbounds for k in 1:i
        H[k,i] = LinearAlgebra.dot(w, Q[k])
        LinearAlgebra.axpy!(-H[k,i],Q[k],w)
    end
end
=#

