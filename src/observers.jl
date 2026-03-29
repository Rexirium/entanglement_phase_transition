mutable struct EntropyObserver{T<:Real} <: AbstractObserver
    """
    Observe and record the entanglement entropy at a specific bond `b` of the MPS during time evolution.
    """
    b::Int
    n::Real
    entropies::Vector{T}
    truncerrs::Vector{T}
    maxbonds::Vector{Int}
    accept::Bool

    EntropyObserver{T}(b::Int; n::Real=1) where T<:Real = new{T}(b, n, T[], T[], Int[], true)
end

mutable struct EntrCorrObserver{lsize, T<:Real} <: AbstractObserver
    b::Int
    n::Real
    op::String
    entrs::Vector{T}
    corrs::Vector{SVector{lsize, T}}
    truncerrs::Vector{T}
    accept::Bool

    EntrCorrObserver{T}(b::Int, lsize::Int; n::Real=1, op::String="Sz") where T<:Real = 
        new{lsize, T}(b, n, op, T[], SVector{lsize, T}[], T[], true)
end

mutable struct EntropyAverager{lsize, T<:Real} <: AbstractObserver
    b::Int
    n::Real
    entr_mean::T
    entr_logm::T
    entr_sstd::T
    accept::Bool

    EntropyAverager{T}(b::Int, lsize::Int; n::Real=1) where T<:Real = 
        new{lsize, T}(b, n, zero(T), zero(T), zero(T), true)
end

mutable struct EntrCorrAverager{lsize, T<:Real} <: AbstractObserver
    b::Int
    n::Real
    op::String
    entr_mean::T
    entr_logm::T
    entr_sstd::T
    corr_mean::SVector{lsize, T}
    corr_sstd::SVector{lsize, T}
    accept::Bool

    EntrCorrAverager{T}(b::Int, lsize::Int; n::Real=1, op::String="Sz") where T<:Real = 
        new{lsize, T}(b, n, op, zero(T), zero(T), zero(T), SVector{lsize}(zeros(T, lsize)), 
        SVector{lsize}(zeros(T, lsize)), true)
end

function mps_monitor!(obs::EntropyObserver, psi::MPS, t::Int, truncerr::Real)
    push!(obs.entropies, ent_entropy(psi, obs.b, obs.n))
    push!(obs.truncerrs, truncerr)
    push!(obs.maxbonds, maxlinkdim(psi))
end

function mps_monitor!(obs::EntrCorrObserver, psi::MPS, t::Int, truncerr::Real)
    push!(obs.entrs, ent_entropy(psi, obs.b, obs.n))
    push!(obs.corrs, correlation_vec(psi, obs.op, obs.op))
    push!(obs.truncerrs, truncerr)
end

function mps_monitor!(obs::EntropyAverager{lsize}, psi::MPS, t::Int, truncerr::Real) where lsize
    """
    Update the mean and SST of entanglement entropy in `obs`.
    Using Welford's algorithm.
    """
    if t > 2 * lsize
        entr = ent_entropy(psi, obs.b, obs.n)
        delta = entr - obs.entr_mean
        delta_log = log(entr) - obs.entr_logm

        obs.entr_mean += delta / (t - 2 * lsize)
        obs.entr_logm += delta_log / (t - 2 * lsize)
        obs.entr_sstd += delta * (entr - obs.entr_mean)
    end
end

function mps_monitor!(obs::EntrCorrAverager{lsize}, psi::MPS, t::Int, truncerr::Real) where lsize
    """
    Update the mean and SST of entanglement entropy and correlation function in `obs`.
    Using Welford's algorithm.
    """
    if t > 2 * lsize
        entr = ent_entropy(psi, obs.b, obs.n)
        corr = correlation_vec(psi, obs.op, obs.op)

        delta_entr = entr - obs.entr_mean
        delta_elog = log(entr) - obs.entr_logm
        delta_corr = corr - obs.corr_mean

        obs.entr_mean += delta_entr / (t - 2 * lsize)
        obs.entr_logm += delta_elog / (t - 2 * lsize)
        obs.corr_mean += delta_corr / (t - 2 * lsize)
        obs.entr_sstd += delta_entr * (entr - obs.entr_mean)
        obs.corr_sstd += delta_corr .* (corr - obs.corr_mean)
    end
end