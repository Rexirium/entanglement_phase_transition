using LinearAlgebra
using ITensors, ITensorMPS

function ITensors.op(::OpName"RdU", ::SiteType"S=1/2", s::Index...; 
    eltype=ComplexF64)
    d = prod(dim.(s))
    M = randn(eltype, d, d)
    Q, _ = NDTensors.qr_positive(M)
    return op(Q, s...)
end

function unitaryGenerator(s::Index...; eltype=ComplexF64)
    d = prod(dim.(s))
    M = randn(eltype, d, d)
    Q, _ = NDTensors.qr_positive(M)
    return op(Q, s1, s2)
end

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

function unitaryGenerator2(s::Index...; eltype=ComplexF64)
    d = prod(dim.(s))
    M = randn(eltype, d, d)
    U = unitarize(M)
    return op(U, s...)
end

function nonHermitianGenerator(eta::Float64, s0::Index)
    M = [eta 0 ; 0 1-eta]
    return op(M, s0)
end

let 
    ss = siteinds("S=1/2", 4)
    psi = random_mps(ss; linkdims=2)
    M = randn(ComplexF64, 2, 2)
    U, _ = NDTensors.qr_positive(M)
    o = ITensors.itensor(U, ss[1]', ss[1])
    matrix(o'*conj(o))
end
