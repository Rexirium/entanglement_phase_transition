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

function ITensors.op(::OpName"WM", ::SiteType"S=1/2", s::Index; x::Real, λ::Real=1.0, Δ::Real=1.0)
    """Create a weak measurement operator for the given site index `s` with parameters `x`, `λ`, and `Δ`."""
    # Assuming `x` is a random variable from a Gaussian distribution
    phiUp = exp(-(x-λ)*(x-λ) / (4*Δ*Δ))
    phiDn = exp(-(x+λ)*(x+λ) / (4*Δ*Δ))
    M = [phiUp 0; 0 phiDn] 
    return op(M, s)
end

function weak_measure!(psi::MPS, loc::Int, para::Tuple{Real, Real}=(1.0, 1.0))
    """Perform a weak measurement on the MPS `psi` at site `loc` with parameters `λ` and `Δ`."""
    (loc <= 0 || loc > length(psi)) && return psi
    # Orthogonalize the MPS at site `loc`
    s = siteind(psi, loc)
    λ, Δ = para
    
    proj = op("ProjUp", s)
    orthogonalize!(psi, loc)
    # Calculate the probability of measuring "Up"
    probUp = real(inner(prime(psi[loc], tags="Site"), proj, psi[loc]))
    p = rand()
    # generate a random variable from a Gaussian distribution
    x = p < probUp ? λ + Δ*randn() : -λ + Δ*randn() 
    M = op("WM", s; x = x, λ = λ, Δ = Δ)
    # Apply the weak measurement operator
    psi = apply(M, psi; cutoff=1e-14)
    normalize!(psi)
end

function mps_evolve(psi0::MPS, ttotal::Int, prob::Real, eta::Real; cutoff::Real=1e-14)
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator applied to pairs of sites,
    and a non-Hermitian operator applied to each site with probability `prob` and parameter `eta`.
    """
    psi = copy(psi0)
    sites = siteinds(psi)
    lsize = length(sites)

    for t in 1:ttotal
        start = isodd(t) ? 1 : 2
        # Apply random unitary operators to pairs of sites
        for j in start:2:lsize-1
            s1, s2 = sites[j], sites[j+1]
            U = op("RdU", s1, s2)
            psi = apply(U, psi; cutoff)
        end
        # Apply non-Hermitian operator to each site with probability `prob` and parameter `eta`
        for j in 1:lsize
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

function mps_evolve(psi0::MPS, ttotal::Int, prob::Real, para::Tuple{Real, Real}; cutoff::Real=1e-14)
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator applied to pairs of sites,
    and a weak measurement operator applied to each site with parameters `λ` and `Δ`.
    """
    psi = copy(psi0)
    sites = siteinds(psi)
    lsize = length(sites)

    for t in 1:ttotal
        start = isodd(t) ? 1 : 2
        # Apply random unitary operators to pairs of sites
        for j in start:2:lsize-1
            s1, s2 = sites[j], sites[j+1]
            U = op("RdU", s1, s2)
            psi = apply(U, psi; cutoff)
        end
        # Apply weak measurement operator to each site with parameters `λ` and `Δ`
        for j in 1:lsize
            p = rand()
            if p < prob
                weak_measure!(psi, j, para)
            end
        end
    end
    return psi
end

function entropy_evolve(psi0::MPS, ttotal::Int, prob::Real, eta::Real, b::Int, which_ent::Real=1; 
     cutoff::Real=1e-14, ent_cutoff::Real=1e-12)
    """
    Same with function `mps_evolve` but with entanglement entropy biparted at site `b` recorded after each time step.
    """
    psi = copy(psi0)
    sites = siteinds(psi) 
    lsize = length(sites)
    # Initialize the entropy vector. 
    entropies = Float64[]
    ini_entropy = Renyi_entropy(psi0, b, which_ent; cutoff=ent_cutoff)
    push!(entropies, ini_entropy)

    for t in 1:ttotal
        start = isodd(t) ? 1 : 2
        for j in start:2:lsize-1
            s1, s2 = sites[j], sites[j+1]
            U = op("RdU", s1, s2)
            psi = apply(U, psi; cutoff)
        end

        for j in 1:lsize
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

function entropy_evolve(psi0::MPS, ttotal::Int, prob::Real, para::Tuple{Real, Real}, b::Int, which_ent::Real=1; 
     cutoff::Real=1e-14, ent_cutoff::Real=1e-12)
    """
    Same with function `mps_evolve` but with entanglement entropy biparted at site `b` recorded after each time step.
    """
    psi = copy(psi0)
    sites = siteinds(psi) 
    lsize = length(sites)
    # Initialize the entropy vector. 
    entropies = Float64[]
    ini_entropy = Renyi_entropy(psi0, b, which_ent; cutoff=ent_cutoff)
    push!(entropies, ini_entropy)

    for t in 1:ttotal
        start = isodd(t) ? 1 : 2
        for j in start:2:lsize-1
            s1, s2 = sites[j], sites[j+1]
            U = op("RdU", s1, s2)
            psi = apply(U, psi; cutoff)
        end

        for j in 1:lsize
            samp = rand()
            if samp < prob
                weak_measure!(psi, j, para)
            end
        end
        entropy = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
        push!(entropies, entropy)
    end
    return psi, entropies
end




