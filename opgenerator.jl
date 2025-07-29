using LinearAlgebra
using ITensors, ITensorMPS

function unitarize(M::AbstractMatrix)
    U = zero(M)
    n = size(M, 2)
    proj = I(n)
    for j in 1:n 
        vec = normalize(proj*M[:,j])
        proj -= vec*vec'
        U[:,j] = vec
    end
    return U
end

function unitaryGenerator(s1::Index, s2::Index)
    M = randn(ComplexF64, 4, 4)
    U = unitarize(M)
    return op(U, s1, s2)
end

function nonHermitianGenerator(eta::Float64, s0::Index)
    M = [eta 0 ; 0 1-eta]
    return op(M, s0)
end
