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

mutable struct EntrCorrObserver{T<:Real} <: AbstractObserver
    b::Int
    n::Real
    op::String
    entrs::Vector{T}
    corrs::Vector{Vector{T}}
    truncerrs::Vector{T}
    accept::Bool

    EntrCorrObserver{T}(b::Int; n::Real=1, op::String="Sz") where T<:Real = 
        new{T}(b, n, op, T[], Vector{T}[], T[], true)
end

mutable struct EntropyAverager{T<:Real} <: AbstractObserver
    b::Int
    n::Real
    tstart::Int
    entr_mean::T
    entr_logm::T
    entr_sstd::T
    accept::Bool

    EntropyAverager{T}(b::Int, tstart::Int; n::Real=1) where T<:Real = 
        new{T}(b, n, tstart, zero(T), zero(T), zero(T), true)
end

mutable struct EntrCorrAverager{T<:Real} <: AbstractObserver
    b::Int
    n::Real
    op::String
    tstart::Int
    entr_mean::T
    entr_logm::T
    entr_sstd::T
    corr_mean::Vector{T}
    corr_sstd::Vector{T}
    accept::Bool

    EntrCorrAverager{T}(b::Int, tstart::Int; n::Real=1, op::String="Sz") where T<:Real = 
        new{T}(b, n, op, tstart, zero(T), zero(T), zero(T), Vector{T}[], Vector{T}[], true)
end

mutable struct EntropyProfile{T <: Real} <: AbstractObserver
    n::Real
    lsize::Int
    entr_distr::Vector{Vector{T}}
    truncerrs::Vector{T}
    accept::Bool

    EntropyProfile{T}(n::Real=1) where T<:Real = new{T}(n, Vector{T}[], T[], true)
end

function mps_record!(obs::EntropyObserver, psi::MPS, t::Int, truncerr::Real)
    push!(obs.entropies, ent_entropy(psi, obs.b, obs.n))
    push!(obs.truncerrs, truncerr)
    push!(obs.maxbonds, maxlinkdim(psi))
end

function mps_record!(obs::EntrCorrObserver, psi::MPS, t::Int, truncerr::Real)
    push!(obs.entrs, ent_entropy(psi, obs.b, obs.n))
    push!(obs.corrs, correlation_vec(psi, obs.op, obs.op))
    push!(obs.truncerrs, truncerr)
end

function mps_record!(obs::EntropyProfile, psi::MPS, t::Int, truncerr::Real)
    entr_distr = [ent_entropy(psi, x, obs.n) for x in 0 : obs.lsize]
    push!(obs.entr_distr, entr_distr)
    push!(obs.truncerrs, truncerr)
end

function mps_record!(obs::EntropyAverager, psi::MPS, t::Int, truncerr::Real)
    """
    Update the mean and SST of entanglement entropy in `obs`.
    Using Welford's algorithm.
    """
    if t > obs.tstart
        entr = ent_entropy(psi, obs.b, obs.n)
        delta = entr - obs.entr_mean
        delta_log = log(entr) - obs.entr_logm

        obs.entr_mean += delta / (t - obs.tstart)
        obs.entr_logm += delta_log / (t - obs.tstart)
        obs.entr_sstd += delta * (entr - obs.entr_mean)
    end
end

function mps_record!(obs::EntrCorrAverager, psi::MPS, t::Int, truncerr::Real)
    """
    Update the mean and SST of entanglement entropy and correlation function in `obs`.
    Using Welford's algorithm.
    """
    tstart = obs.tstart
    if t > tstart
        entr = ent_entropy(psi, obs.b, obs.n)
        corr = correlation_vec(psi, obs.op, obs.op)

        delta_entr = entr - obs.entr_mean
        delta_elog = log(entr) - obs.entr_logm
        delta_corr = corr - obs.corr_mean

        obs.entr_mean += delta_entr / (t - tstart)
        obs.entr_logm += delta_elog / (t - tstart)
        obs.corr_mean += delta_corr / (t - tstart)
        obs.entr_sstd += delta_entr * (entr - obs.entr_mean)
        obs.corr_sstd += delta_corr .* (corr - obs.corr_mean)
    end
end