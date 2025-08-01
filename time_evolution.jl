using LinearAlgebra
using ITensors, ITensorMPS

function ITensors.op(::OpName"RdU", ::SiteType"S=1/2", s::Index...; eltype=ComplexF64)
    d = prod(dim.(s))
    M = randn(eltype, d, d)
    Q, _ = NDTensors.qr_positive(M)
    return op(Q, s...)
end

function ITensors.op(::OpName"NH", ::SiteType"S=1/2", s::Index; eta::Real)
    return op([1 0; 0 eta], s)
end

function unitaryGenerator(s::Index...; eltype=ComplexF64)
    d = prod(dim.(s))
    M = randn(eltype, d, d)
    Q, _ = NDTensors.qr_positive(M)
    return op(Q, s1, s2)
end

function nonHermitianGenerator(eta::Float64, s0::Index)
    M = [1 0 ; 0 eta]
    return op(M, s0)
end

function mps_evolve(psi0::MPS, time::Int, prob::Real, eta::Real; cutoff::Real=1e-14)
    psi = copy(psi0)
    sites = siteinds(psi)
    Ls= length(sites)
    for t in 1:time
        start = isodd(t) ? 1 : 2
        for j in start:2:Ls-1
            s1, s2 = sites[j], sites[j+1]
            U = op("RdU", s1, s2)
            psi = apply(U, psi; cutoff)
        end
        for j in 1:Ls
            p = rand()
            if p < prob
                s = sites[j]
                M = op("NH", s; eta=eta)
                psi = apply(M, psi; cutoff)
            end
        end
        normalize!(psi)
    end
    return psi
end

let 
    ss = siteinds("S=1/2", 4)
    psi = random_mps(ss; linkdims=2)
    o = op("NH", ss[1]; eta=0.5)
    array(o)
end
