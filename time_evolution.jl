using LinearAlgebra
using ITensors, ITensorMPS
include("entanglement_entropies.jl")

function ITensors.op(::OpName"RdU", ::SiteType"S=1/2", s::Index...; eltype=ComplexF64)
    """
    Create a random unitary operator for the given site indices `s`.
    """
    d = prod(dim.(s))
    M = randn(eltype, d, d)
    Q, _ = NDTensors.qr_positive(M)
    return op(Q, s...)
end

function ITensors.op(::OpName"NH", ::SiteType"S=1/2", s::Index; eta::Real)
    """
    Create a non-Hermitian operator for the given site index `s` with parameter `eta`.
    """
    return op([1 0; 0 eta], s)
end

function ITensors.op(::OpName"WM", ::SiteType"S=1/2", s::Index, x::Real, λ::Real=1.0, Δ::Real=1.0)
    """Create a weak measurement operator for the given site index `s` with parameters `x`, `λ`, and `Δ`."""
    # Assuming `x` is a random variable from a Gaussian distribution
    phiUp = exp(-(x-λ)*(x-λ) / (4*Δ*Δ))
    phiDown = exp(-(x+λ)*(x+λ) / (4*Δ*Δ))
    M = [phiUp 0; 0 phiDown] 
    return op(M, s)
end

function weak_measure!(psi::MPS, loc::Int, λ::Real=1.0, Δ::Real=1.0)
    """Perform a weak measurement on the MPS `psi` at site `loc` with parameters `λ` and `Δ`."""
    (loc <= 0 || loc > length(psi)) && return psi
    # Orthogonalize the MPS at site `loc`
    s = siteind(psi, loc)
    proj = op("ProjUp", s)
    orthogonalize!(psi, loc)
    # Calculate the probability of measuring "Up"
    probUp = real(inner(prime(psi[loc], tags="Site"), proj, psi[loc]))
    p = rand()
    # generate a random variable from a Gaussian distribution
    x = p < probUp ? λ + Δ*randn() : -λ + Δ*randn() 
    M = op("WM", s, x, λ, Δ)
    # Apply the weak measurement operator
    psi = apply(M, psi; cutoff=1e-14)
    normalize!(psi)
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

function mps_evolve(psi0::MPS, ttotal::Int, prob::Real, eta::Real; cutoff::Real=1e-14)
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator applied to pairs of sites,
    and a non-Hermitian operator applied to each site with probability `prob` and parameter `eta`.
    """
    psi = copy(psi0)
    sites = siteinds(psi)
    Lsize = length(sites)

    for t in 1:ttotal
        start = isodd(t) ? 1 : 2
        # Apply random unitary operators to pairs of sites
        for j in start:2:Lsize-1
            s1, s2 = sites[j], sites[j+1]
            U = op("RdU", s1, s2)
            psi = apply(U, psi; cutoff)
        end
        # Apply non-Hermitian operator to each site with probability `prob` and parameter `eta`
        for j in 1:Lsize
            p = rand()
            if p < prob
                s = sites[j]
                M = op("NH", s; eta=eta)
                psi = apply(M, psi; cutoff)
                # Normalize the MPS after applying the non Hermitian operator
                normalize!(psi)
            end
        end
    end
    return psi
end

function entropy_evolve(psi0::MPS, ttotal::Int, prob::Real, eta::Real, b::Int, which_ent::Real; 
     cutoff::Real=1e-14, ent_cutoff::Real=1e-12)
    """
    Same with function `mps_evolve` but with entanglement entropy biparted at site `b` recorded after each time step.
    """
    psi = copy(psi0)
    sites = siteinds(psi) 
    Lsize = length(sites)
    # Initialize the entropy vector. 
    entropies = Float64[]
    ini_entropy = Renyi_entropy(psi0, b, which_ent; cutoff=ent_cutoff)
    push!(entropies, ini_entropy)

    for t in 1:ttotal
        start = isodd(t) ? 1 : 2
        for j in start:2:Lsize-1
            s1, s2 = sites[j], sites[j+1]
            U = op("RdU", s1, s2)
            psi = apply(U, psi; cutoff)
        end

        for j in 1:Lsize
            samp = rand()
            if samp < prob
                s = sites[j]
                M = op("NH", s; eta=eta)
                psi = apply(M, psi; cutoff)
                normalize!(psi)
            end
        end
        entropy = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
        push!(entropies, entropy)
    end
    return psi, entropies
end
#=
let 
    L = 8
    T = 2L
    b = L÷2
    p, eta = 0.5, 0.5
    ss = siteinds("S=1/2", L)
    psi0 = random_mps(ss; linkdims = 4)
    [Renyi_entropy(psi0, x, 1) for x in 0:L ]
end
=#


