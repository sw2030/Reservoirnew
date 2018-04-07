function stencilgmres(A, b::Grid, restrt::Int64; tol::Real=1e-5, maxiter::Int=200, ifprint=false, M=identity, x = zero(b))
    realn = gridsize(b)
    bnrm2 = norm(b)
    if bnrm2==0 bnrm2 = 1.0 end
    r = zero(b)+b
    A_mul_B!(-1.0, A, x, 1.0, r)
    #r = b-A*x
    err = norm(r)/bnrm2
    iter = 0
    ismax = false
    if err<tol return x, ismax, iter, err end

    restrt=min(restrt, realn-1)
    Q = [zero(b) for i in 1:restrt+1]
    H = zeros(restrt+1, restrt)
    cs = zeros(restrt)
    sn = zeros(restrt)
    s = zeros(restrt+1)
    flag = -1
    i = 1

    y = zeros(restrt+1)

    for iter in 1:maxiter
        if ifprint print(iter) end
        r = Q[1]
        copy!(r, b)
        A_mul_B!(-1.0, A, x, 1.0, r)
        copy!(r, M(r))
        fill!(s, 0)
        s[1] = norm(r)
        scale!(r, inv(s[1]))

        for i in 1:restrt
            w = Q[i+1]
            A_mul_B!(1.0, A, Q[i], 0.0, w)
            copy!(w, M(w))
            #w = A*Q[i]
            for k in 1:i
                H[k,i] = dot(w, Q[k])
                LinAlg.axpy!(-H[k,i],Q[k],w)
            end
            H[i+1,i] = norm(w)
            scale!(w, inv(H[i+1,i]))
            #Q[i+1] = w/H[i+1, i]
            for k in 1:i-1
                temp     =  cs[k]*H[k,i] + sn[k]*H[k+1,i]
                H[k+1,i] = -sn[k]*H[k,i] + cs[k]*H[k+1,i]
                H[k,i]   = temp
            end

            cs[i], sn[i] = LinAlg.givensAlgorithm(H[i, i], H[i+1, i])
            s[i+1] = -sn[i]*s[i]
            s[i]   = cs[i]*s[i]
            H[i,i] = cs[i]*H[i,i] + sn[i]*H[i+1,i]
            H[i+1,i] = 0.0
            err  = norm(s[i+1])/bnrm2
            #@show err

            if err <= tol
                #y[1:i]  = H[1:i,1:i] \ s[1:i]
                copy!(y, s)
                A_ldiv_B!(UpperTriangular(view(H, 1:i, 1:i)), view(y, 1:i))
                for k in 1:i
                    #x += y[k]*Q[k]
                    LinAlg.axpy!(y[k],Q[k],x)
                end
                # x += Q[:,1:i]*y
                flag = 0; break
            end
        end
        if  err <= tol
            flag = 0
            break
        end
        #y = H[1:restrt,1:restrt]\s[1:restrt]
        copy!(y, s)
        A_ldiv_B!(UpperTriangular(view(H, 1:restrt, 1:restrt)), view(y, 1:restrt))  #x += Q[:,1:restrt]*y
        for k in 1:restrt
            LinAlg.axpy!(y[k],Q[k],x)  #x += y[k]*Q[k]
        end
        copy!(r, b)
        A_mul_B!(-1.0, A, x, 1.0, r)
        copy!(r, M(r))
        s[i+1] = norm(r)
        err = s[i+1]/bnrm2
        if err<=tol
            flag = 0
            break
        end

        #r = b-A*x
    end
    if flag==-1
        print("Maxiter")
        ismax = true
    end

    return x, ismax, iter, err
end