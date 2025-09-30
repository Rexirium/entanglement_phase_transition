using ITensors, ITensorMPS
include("entanglement_entropies.jl")

function ITensors.op(::OpName"RdU", ::SiteType"S=1/2", s::Index...; eltype=ComplexF64)
    """
    Create a random unitary operator for the given site indices `s`.
    """
    M = randn(eltype, 4, 4)
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
    samp = rand()
    # generate a random variable from a Gaussian distribution
    x = samp < probUp ? λ + Δ*randn() : -λ + Δ*randn() 
    M = op("WM", s; x = x, λ = λ, Δ = Δ)
    # Apply the weak measurement operator
    apply!(M, psi, loc)
    normalize!(psi)
end

function apply!(G1::ITensor, psi::MPS, loc::Int)
    """
    Apply the gate `G1` to the MPS `psi` at site `loc` inplace.
    """
    orthogonalize!(psi, loc)
    A = noprime(psi[loc] * G1)
    psi[loc] = A
end

function apply!(G2::ITensor, psi::MPS, j1::Int, j2::Int; cutoff::Real=1e-12)
    """
    Apply two adjacent site gate `G2` to the MPS `psi` at sites `loc` inplace.
    """
    (j2 - j1 != 1) && error("The two sites is not adjacent or in wrong order!")
    orthogonalize!(psi, j1)
    A = psi[j1] * psi[j2] * G2
    noprime!(A)
    if j1 == 1
        psi[j1], S, psi[j2] = svd(A, siteind(psi, j1); cutoff=cutoff)
    else
        psi[j1], S, psi[j2] = svd(A, (siteind(psi, j1), linkind(psi, j1-1)); cutoff=cutoff)
    end
    psi[j2] *= S
    set_ortho_lims!(psi, j2:j2)
end

function mps_evolve(psi0::MPS, ttotal::Int, prob::Real, eta::Real; cutoff::Real=1e-12)
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator applied to pairs of sites,
    and a non-Hermitian operator applied to each site with probability `prob` and parameter `eta`.
    """
    psi = copy(psi0)
    sites = siteinds(psi)
    lsize = length(sites)

    for t in 1:ttotal
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1])
            apply!(U, psi, j, j+1; cutoff=cutoff)
        end
        # Apply non-Hermitian operator to each site with probability `prob` and parameter `eta`
        for j in 1:lsize
            samp = rand()
            if samp < prob
                M = op("NH", sites[j]; eta=eta)
                apply!(M, psi, j)
                # Normalize the MPS after applying the non Hermitian operator
                normalize!(psi)
            end
        end
    end
    return psi
end

function mps_evolve!(psi::MPS, ttotal::Int, prob::Real, eta::Real; cutoff::Real=1e-12)
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator applied to pairs of sites,
    and a non-Hermitian operator applied to each site with probability `prob` and parameter `eta`. (inplace version)
    """
    sites = siteinds(psi)
    lsize = length(sites)

    for t in 1:ttotal
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1])
            apply!(U, psi, j, j+1; cutoff=cutoff)
        end
        # Apply non-Hermitian operator to each site with probability `prob` and parameter `eta`
        for j in 1:lsize
            samp = rand()
            if samp < prob
                M = op("NH", sites[j]; eta=eta)
                apply!(M, psi, j)
                # Normalize the MPS after applying the non Hermitian operator
                normalize!(psi)
            end
        end
    end
end

function mps_evolve(psi0::MPS, ttotal::Int, prob::Real, para::Tuple{Real, Real}; cutoff::Real=1e-12)
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator applied to pairs of sites,
    and a weak measurement operator applied to each site with parameters `λ` and `Δ`.
    """
    psi = copy(psi0)
    sites = siteinds(psi)
    lsize = length(sites)

    for t in 1:ttotal
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1])
            apply!(U, psi, j, j+1; cutoff=cutoff)
        end
        # Apply weak measurement operator to each site with parameters `λ` and `Δ`
        for j in 1:lsize
            samp = rand()
            if samp < prob
                weak_measure!(psi, j, para)
            end
        end
    end
    return psi
end

function mps_evolve!(psi::MPS, ttotal::Int, prob::Real, para::Tuple{Real, Real}; cutoff::Real=1e-12)
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator applied to pairs of sites,
    and a weak measurement operator applied to each site with parameters `λ` and `Δ`. (inplace version)
    """
    sites = siteinds(psi)
    lsize = length(sites)

    for t in 1:ttotal
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1])
            apply!(U, psi, j, j+1; cutoff=cutoff)
        end
        # Apply weak measurement operator to each site with parameters `λ` and `Δ`
        for j in 1:lsize
            samp = rand()
            if samp < prob
                weak_measure!(psi, j, para)
            end
        end
    end
end

function entropy_evolve(psi0::MPS, ttotal::Int, prob::Real, eta::Real, b::Int, which_ent::Real=1; 
     cutoff::Real=1e-12, ent_cutoff::Real=1e-10)
    """
    Same with function `mps_evolve` but with entanglement entropy biparted at site `b` recorded after each time step.
    """
    psi = copy(psi0)
    sites = siteinds(psi) 
    lsize = length(sites)
    # Initialize the entropy vector. 
    entropies = zeros(Float64, ttotal+1)
    entropies[1] = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)

    for t in 1:ttotal
        # the layer for random unitary operators
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1])
            apply!(U, psi, j, j+1; cutoff=cutoff)
        end
        # the layer for random non-Hermitian gates
        for j in 1:lsize
            samp = rand()
            if samp < prob
                M = op("NH", sites[j]; eta=eta)
                apply!(M, psi, j)
                normalize!(psi)
            end
        end
        # Record the entanglement entropy after each time step
        entropies[t+1] = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
    end
    return psi, entropies
end

function entropy_evolve!(psi::MPS, ttotal::Int, prob::Real, eta::Real, b::Int, which_ent::Real=1; 
     cutoff::Real=1e-12, ent_cutoff::Real=1e-10)
    """
    Same with function `mps_evolve!` but with entanglement entropy biparted at site `b` recorded after each time step.
    """
    sites = siteinds(psi) 
    lsize = length(sites)
    # Initialize the entropy vector. 
    entropies = zeros(Float64, ttotal+1)
    entropies[1] = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)

    for t in 1:ttotal
        # the layer for random unitary operators
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1])
            apply!(U, psi, j, j+1; cutoff=cutoff)
        end
        # the layer for random non-Hermitian gates
        for j in 1:lsize
            samp = rand()
            if samp < prob
                M = op("NH", sites[j]; eta=eta)
                apply!(M, psi, j)
                normalize!(psi)
            end
        end
        # Record the entanglement entropy after each time step
        entropies[t+1] = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
    end
    return entropies
end

function entropy_evolve(psi0::MPS, ttotal::Int, prob::Real, para::Tuple{Real, Real}, b::Int, which_ent::Real=1; 
     cutoff::Real=1e-12, ent_cutoff::Real=1e-10)
    """
    Same with function `mps_evolve` but with entanglement entropy biparted at site `b` recorded after each time step.
    """
    psi = copy(psi0)
    sites = siteinds(psi) 
    lsize = length(sites)
    # Initialize the entropy vector. 
    entropies = zeros(Float64, ttotal+1)
    entropies[1] = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)

    for t in 1:ttotal
        # Initialize the entropy vector.
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1])
            apply!(U, psi, j, j+1; cutoff=cutoff)
        end
        # apply random weak measurement
        for j in 1:lsize
            samp = rand()
            if samp < prob
                weak_measure!(psi, j, para)
            end
        end
        # Record the entanglement entropy after each time step
        entropies[t+1] = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
    end
    return psi, entropies
end

function entropy_evolve!(psi::MPS, ttotal::Int, prob::Real, para::Tuple{Real, Real}, b::Int, which_ent::Real=1; 
     cutoff::Real=1e-12, ent_cutoff::Real=1e-10)
    """
    Same with function `mps_evolve!` but with entanglement entropy biparted at site `b` recorded after each time step.
    """
    sites = siteinds(psi) 
    lsize = length(sites)
    # Initialize the entropy vector. 
    entropies = zeros(Float64, ttotal+1)
    entropies[1] = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)

    for t in 1:ttotal
        # Initialize the entropy vector.
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1])
            apply!(U, psi, j, j+1; cutoff=cutoff)
        end
        # apply random weak measurement
        for j in 1:lsize
            samp = rand()
            if samp < prob
                weak_measure!(psi, j, para)
            end
        end
        # Record the entanglement entropy after each time step
        entropies[t+1] = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
    end
    return entropies
end

if abspath(PROGRAM_FILE) == @__FILE__ 
# let
    ss = siteinds("S=1/2", 10)
    psi = random_mps(ss; linkdims = 4)

    U = op("RdU", ss[3], ss[4])
    apply!(U, psi, 3, 4; cutoff=1e-12)
    ortho_lims(psi)
end
