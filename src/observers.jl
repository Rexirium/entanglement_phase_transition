mutable struct EntropyObserver{T<:Real} <: AbstractObserver
    """
    Observe and record the entanglement entropy at a specific bond `b` of the MPS during time evolution.
    """
    b::Int
    n::Real
    entropies::Vector{T}
    truncerrs::Vector{T}
    maxbonds::Vector{Int}

    EntropyObserver{T}(b::Int; n::Real=1) where T<:Real = new{T}(b, n, T[], T[], Int[])
end

mutable struct EntrCorrObserver{T<:Real} <: AbstractObserver
    b::Int
    len::Int
    n::Real
    op::String
    entrs::Vector{T}
    corrs::Vector{Vector{T}}
    truncerrs::Vector{T}

    EntrCorrObserver{T}(b::Int, len::Int; n::Real=1, op::String="Sz") where T<:Real = 
        new{T}(b, len, n, op, T[], Vector{T}[], T[])
end

mutable struct EntropyAverager{T<:Real} <: AbstractObserver
    b::Int
    len::Int
    n::Real
    entr_mean::T
    entr_sstd::T
    accept::Bool

    EntropyAverager{T}(b::Int, len::Int; n::Real=1) where T<:Real = 
        new{T}(b, len, n, zero(T), zero(T), true)
end

mutable struct EntrCorrAverager{T<:Real} <: AbstractObserver
    b::Int
    len::Int
    n::Real
    op::String
    entr_mean::T
    entr_sstd::T
    corr_mean::Vector{T}
    corr_sstd::Vector{T}
    accept::Bool

    EntrCorrAverager{T}(b::Int, len::Int; n::Real=1, op::String="Sz") where T<:Real = 
        new{T}(b, len, n, op, zero(T), zero(T), 
        zeros(T, len), zeros(T, len), true)
end

function mps_monitor!(obs::EntropyObserver{T}, psi::MPS, t::Int, truncerr::Real) where T<:Real
    push!(obs.entropies, ent_entropy(psi, obs.b, obs.n))
    push!(obs.truncerrs, truncerr)
    push!(obs.maxbonds, maxlinkdim(psi))
end

function mps_monitor!(obs::EntrCorrObserver{T}, psi::MPS, t::Int, truncerr::Real) where T<:Real
    push!(obs.entrs, ent_entropy(psi, obs.b, obs.n))
    push!(obs.corrs, correlation_vec(psi, obs.op, obs.op))
    push!(obs.truncerrs, truncerr)
end

function mps_monitor!(obs::EntropyAverager{T}, psi::MPS, t::Int, truncerr::Real) where T<:Real
    """
    Update the mean and SST of entanglement entropy in `obs`.
    Using Welford's algorithm.
    """
    if t > 2 * obs.len
        entr = ent_entropy(psi, obs.b, obs.n)
        delta = entr - obs.entr_mean
        obs.entr_mean += delta / (t - 2 * obs.len)
        obs.entr_sstd += delta * (entr - obs.entr_mean)
    end
end

function mps_monitor!(obs::EntrCorrAverager{T}, psi::MPS, t::Int, truncerr::Real) where T<:Real
    """
    Update the mean and SST of entanglement entropy and correlation function in `obs`.
    Using Welford's algorithm.
    """
    if t > 2 * obs.len
        entr = ent_entropy(psi, obs.b, obs.n)
        corr = correlation_vec(psi, obs.op, obs.op)

        delta_entr = entr - obs.entr_mean
        delta_corr = corr .- obs.corr_mean
        obs.entr_mean += delta_entr / (t - 2 * obs.len)
        obs.corr_mean .+= delta_corr ./ (t - 2 * obs.len)
        obs.entr_sstd += delta_entr * (entr - obs.entr_mean)
        obs.corr_sstd .+= delta_corr .* (corr .- obs.corr_mean)
    end
end