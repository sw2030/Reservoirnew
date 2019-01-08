using CuArrays, CUDAnative, LinearAlgebra

const GPUGrid{T,N,P}    = Reservoir.Grid{T,N,P,CuArrays.CuArray{T,N}}
const GPUStencil{T,N,P} = Reservoir.Stencil{T,N,P,CuArrays.CuArray{Reservoir.StencilPoint{T,N,P},N}}
const GPUMGrid{M,T,N,P} = Reservoir.MGrid{M,T,N,P,<:CuArrays.CuArray}
const GPUMStencil{MM,T,N,P} = Reservoir.MStencil{MM,T,N,P,<:CuArrays.CuArray}
    

LinearAlgebra.axpy!(a::Number, g1::GPUMGrid, g2::GPUMGrid) = LinearAlgebra.axpy!.(a, g1, g2)
function mul!(ry::GPUGrid{T,3,P}, rA::GPUStencil{T,3,P}, rx::GPUGrid{T,3,P}) where {T,P}
    
    Sidx = Reservoir.Sindx{3,P}()()   
    function kernel(y, A, x, Sid)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        j = (blockIdx().y-1) * blockDim().y + threadIdx().y
        k = (blockIdx().z-1) * blockDim().z + threadIdx().z
        
        if i <= size(A,1) && j <= size(A,2) && k <= size(A,3)
            tmp = zero(T)
            Sijk = A[i,j,k].value
            @inbounds for c in 1:P 
                tmp += Sijk[c] * x[i+Sid[c][1]+1, j+Sid[c][2]+1, k+Sid[c][3]+1]
            end
            y[i+1,j+1,k+1] += tmp
        end

        return
    end
    
    max_threads = 256
    threads_x   = min(max_threads, size(ry,1))
    threads_y   = min(max_threads ÷ threads_x, size(ry,2))
    threads_z   = min(max_threads ÷ threads_x ÷ threads_y, size(ry,3))
    threads     = (threads_x, threads_y, threads_z)
    blocks      = ceil.(Int, (size(ry,1), size(ry,2), size(ry,3)) ./ threads)
    
    @cuda threads=threads blocks=blocks kernel(ry.A, rA.v, rx.A, Sidx)
    
    ry
end
function gemv!(a::Number, rA::GPUStencil{T,3,P}, rx::GPUGrid{T,3,P}, b::Number, ry::GPUGrid{T,3,P}) where {T,P}
    
    Sidx = Reservoir.Sindx{3,P}()() 
    function kernel(a, A, x, b, y, Sid)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        j = (blockIdx().y-1) * blockDim().y + threadIdx().y
        k = (blockIdx().z-1) * blockDim().z + threadIdx().z
        
        if i <= size(y,1)-2 && j <= size(y,2)-2 && k <= size(y,3)-2
            if b!=1
                y[i+1, j+1, k+1] *= b
            end
            tmp = zero(T)
            Sijk = A[i,j,k].value
            @inbounds for c in 1:P 
                tmp += Sijk[c] * x[i+Sid[c][1]+1, j+Sid[c][2]+1, k+Sid[c][3]+1]
            end
            y[i+1,j+1,k+1] += a*tmp
        end

        return
    end
    
    max_threads = 256
    threads_x   = min(max_threads, size(ry,1))
    threads_y   = min(max_threads ÷ threads_x, size(ry,2))
    threads_z   = min(max_threads ÷ threads_x ÷ threads_y, size(ry,3))
    threads     = (threads_x, threads_y, threads_z)
    blocks      = ceil.(Int, (size(ry,1), size(ry,2), size(ry,3)) ./ threads)
    
    @cuda threads=threads blocks=blocks kernel(a, rA.v, rx.A, b, ry.A, Sidx)
    
    ry
end
function make_P_E_precond_1(MS::GPUMStencil{4,Float64,3,7})    
    nx, ny, nz = size(MS[1])
    ps = [copy(MS.stencils[i].v) for i in 1:4]
    es = [copy(MS.stencils[i].v) for i in 1:4]
    
    function kernel(p1, p2, p3, p4, e1, e2, e3, e4)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        j = (blockIdx().y-1) * blockDim().y + threadIdx().y
        k = (blockIdx().z-1) * blockDim().z + threadIdx().z
        
        if i<=size(p1,1) && j<=size(p1,2) && k<=size(p1,3)
            e1[i,j,k] = Reservoir.StencilPoint{Float64,3,7}(Base.setindex(e1[i,j,k].value,0.0,4))
            e2[i,j,k] = Reservoir.StencilPoint{Float64,3,7}(Base.setindex(e2[i,j,k].value,0.0,4))
            e3[i,j,k] = Reservoir.StencilPoint{Float64,3,7}(Base.setindex(e3[i,j,k].value,0.0,4))
            e4[i,j,k] = Reservoir.StencilPoint{Float64,3,7}(Base.setindex(e4[i,j,k].value,0.0,4))
            d = p1[i,j,k].value[4]*p4[i,j,k].value[4] - p2[i,j,k].value[4]*p3[i,j,k].value[4]
            v1 = p4[i,j,k].value[4]/d
            v2 = -p3[i,j,k].value[4]/d
            v3 = -p2[i,j,k].value[4]/d
            v4 = p1[i,j,k].value[4]/d
            z = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            p1[i,j,k] = Reservoir.StencilPoint{Float64,3,7}(Base.setindex(z,v1,4))
            p2[i,j,k] = Reservoir.StencilPoint{Float64,3,7}(Base.setindex(z,v3,4))
            p3[i,j,k] = Reservoir.StencilPoint{Float64,3,7}(Base.setindex(z,v2,4))
            p4[i,j,k] = Reservoir.StencilPoint{Float64,3,7}(Base.setindex(z,v4,4))
        end
    end
    
    max_threads = 256
    threads_x   = min(max_threads, nx)
    threads_y   = min(max_threads ÷ threads_x, ny)
    threads_z   = min(max_threads ÷ threads_x ÷ threads_y, nz)
    threads     = (threads_x, threads_y, threads_z)
    blocks      = ceil.(Int, (nx, ny, nz) ./ threads)
    
    @cuda threads=threads blocks=blocks kernel(ps[1], ps[2], ps[3], ps[4], es[1], es[2], es[3], es[4])
    
    Pinv = MStencil{4,Float64,3,7,typeof(ps[1])}(Tuple(Stencil{Float64,3,7,typeof(ps[1])}.(ps)))
    E    = MStencil{4,Float64,3,7,typeof(ps[1])}(Tuple(Stencil{Float64,3,7,typeof(ps[1])}.(es)))
    
    Pinv, E
end
     
LinearAlgebra.norm(g::GPUGrid) = norm(g.A)
LinearAlgebra.dot(g1::GPUGrid{T,3,P}, g2::GPUGrid{T,3,P}) where {T,P} = LinearAlgebra.dot(g1.A, g2.A)

    