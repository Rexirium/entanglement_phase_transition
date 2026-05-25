function ITensors.op(::OpName"RdU", ::SiteType"S=1/2", s::Index...; eltype::DataType=ComplexF64)
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
    M = diagm(shuffle([one(eta), eta]))
    return op(M, s)
end

function ITensors.op(::OpName"WM", ::SiteType"S=1/2", s::Index; x::Real, λ::Real=1.0, Δ::Real=1.0)
    """Create a weak measurement operator for the given site index `s` with parameters `x`, `λ`, and `Δ`."""
    # Assuming `x` is a random variable from a Gaussian distribution
    phiUp = exp(-((x - λ) / (2*Δ)) ^ 2)
    phiDn = exp(-((x + λ) / (2*Δ)) ^ 2)
    return op([phiUp 0; 0 phiDn], s)
end

function proj_measure!(psi::MPS, loc::Int)
    (loc <= 0 || loc > length(psi)) && return psi
    # Orthogonalize the MPS at site `loc`
    orthogonalize!(psi, loc)
    # Calculate the probability of measuring "Up"
    s = siteind(psi, loc)
    proj = ITensor(s', s)
    proj[s'=>1, s=>1] = 1.0

    Aup = noprime(psi[loc] * proj)
    probUp = real(scalar(Aup * dag(psi[loc])))
    # Collapse the states
    if rand() < probUp
        psi[loc] = Aup
        normalize!(psi)
        return true
    else
        proj[s'=>1, s=>1] = 0.0
        proj[s'=>2, s=>2] = 1.0

        psi[loc] *= proj
        noprime!(psi[loc])
        normalize!(psi)
        return false
    end
end

function proj_measure!(psi::MPS, phi::MPS, loc::Int)
    (loc <= 0 || loc > length(psi)) && return psi
    # Orthogonalize the MPS at site `loc`
    orthogonalize!(psi, loc)
    orthogonalize!(phi, loc)
    # Calculate the probability of measuring "Up"
    s = siteind(psi, loc)
    proj = ITensor(s', s)
    proj[s'=>1, s=>1] = 1.0

    Aup = noprime(psi[loc] * proj)
    probUp = real(scalar(Aup * dag(psi[loc])))
    # Collapse the states
    if rand() < probUp
        psi[loc] = Aup
        normalize!(psi)
        phi[loc] *= proj
        noprime!(phi[loc])
        normalize!(phi)
        return true
    else
        proj[s'=>1, s=>1] = 0.0
        proj[s'=>2, s=>2] = 1.0

        psi[loc] *= proj
        noprime!(psi[loc])
        normalize!(psi)
        phi[loc] *= proj
        noprime!(phi[loc])
        normalize!(phi)
        return false
    end
end

function weak_measure!(psi::MPS, loc::Int, para::Tuple{T, T}=(1.0, 1.0)) where T<:Real
    """Perform a weak measurement on the MPS `psi` at site `loc` with parameters `λ` and `Δ`."""
    (loc <= 0 || loc > length(psi)) && return psi
    # Orthogonalize the MPS at site `loc`
    orthogonalize!(psi, loc)
    s = siteind(psi, loc)
    λ, Δ = para
    
    proj = ITensor(s)
    proj[s => 1] = 1.0
    Aup = psi[loc] * dag(proj)
    # Calculate the probability of measuring "Up"
    probUp = real(scalar(dag(Aup) * Aup))
    # generate a random variable from a Gaussian distribution
    x = rand(T) < probUp ? λ + Δ*randn(T) : -λ + Δ*randn(T)
    M = op("WM", s; x = x, λ = λ, Δ = Δ)
    # Apply the weak measurement operator
    psi[loc] *= M
    noprime!(psi[loc])
    normalize!(psi)
    return x
end

########################## Monitors #######################

abstract type AbstractMonitor end

struct NHMonitor{Tp <: AbstractFloat} <: AbstractMonitor
    """
    Store the parameters for the non-Hermitian disentangler. 
    `prob` is the probability of applying the non-Hermitian operator, 
    and `eta` is the parameter of the non-Hermitian operator.
    """
    prob::Tp
    eta::Tp
end

struct PMMonitor{Tp <: AbstractFloat} <: AbstractMonitor
    """
    Store the parameters for the projective measurement disentangler.
    """
    probs::Vector{Tp}

    PMMonitor{Tp}(lsize::Int, n::Real) where Tp<:AbstractFloat = new{Tp}(rand(Tp, lsize) .^ n)
    PMMonitor(rxs::Vector{Tp}, n::Real) where Tp<:AbstractFloat = new{Tp}(rxs .^ n)
    PMMonitor(prob::Tp, lsize::Int) where Tp<:AbstractFloat = new{Tp}(fill(prob, lsize))
end

function monitor!(psi::MPS, mnt::NHMonitor{Tp}) where Tp<:AbstractFloat
    """
    Apply the non-Hermitian disentangler to the MPS `psi` inplace.
    """
    for j in length(psi):-1:1
        if rand() < mnt.prob
            M = op("NH", siteind(psi, j); eta=mnt.eta)
            applyn!(M, psi, j)
            normalize!(psi)
        end
    end
    return zero(Tp)
end

function monitor!(psi::MPS, mnt::PMMonitor{Tp}) where Tp<:AbstractFloat
    """
    Apply the projective measurement disentangler to the MPS `psi` inplace.
    """
    for j in length(psi):-1:1
        if rand() < mnt.probs[j]
            proj_measure!(psi, j)
        end
    end
    return zero(Tp)
end

function monitor!(psi::MPS, phi::MPS, mnt::NHMonitor{Tp}) where Tp<:AbstractFloat
    """
    Apply the non-Hermitian disentangler to the MPS `psi` inplace, and apply the same operation to `phi`.
    """
    for j in length(psi):-1:1
        if rand() < mnt.prob
            M = op("NH", siteind(psi, j); eta=mnt.eta)
            applyn!(M, psi, j)
            applyn!(M, phi, j)
            normalize!(psi)
            normalize!(phi)
        end
    end
    return zero(Tp)
end

function monitor!(psi::MPS, phi::MPS, mnt::PMMonitor{Tp}) where Tp<:AbstractFloat
    """
    Apply the projective measurement disentangler to the MPS `psi` inplace, and apply the same operation to `phi`.
    """
    for j in length(psi):-1:1
        if rand() < mnt.probs[j]
            proj_measure!(psi, phi, j)
        end
    end
    return zero(Tp)
end


