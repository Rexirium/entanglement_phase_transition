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
            apply1!(M, psi, j)
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
            apply1!(M, psi, j)
            apply1!(M, phi, j)
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
