module myprofile

using CuArrays, NVTX, LinearAlgebra

function foo(n)
        A = CuArrays.CuArray(randn(1000, 2000))
	A2 = CuArray(randn(1000, 2000))
	ii = 1
        ii = innerloop(A, n, ii, A2)
	A[1,1] += 1.0
end
function innerloop(A, n, ii, A2)
        for j in 1:n
                NVTX.@range string(j, "-", 1) norm(A)
                NVTX.@range string(j, "-", 2) norm(A)
		NVTX.@range string(j, "-A2") norm(A2)
		A[1,1] += 1.0
	end
	
        return ii+1
end

end
