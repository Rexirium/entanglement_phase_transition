struct CalcResult{T<:Real}
    entr_mean::T
    entr_sem::T
    corr_mean::Vector{T}
    corr_sem::Vector{T}
end

abstract type AbstractResult end

mutable struct EntropyResults{T<:Real} <: AbstractResult
    """
    Store the values of entanglement entropy over multiple samples after time evolution.
    """
    type::DataType
    b::Int
    n::Real
    nsamp::Int
    entropies::Vector{T}
    EntropyResults{T}(b::Int; n=1, nsamp::Int = 100) where T<:Real = 
        new{T}(T, b, n, nsamp, Vector{T}(undef, nsamp))
end

mutable struct EntrCorrResults{lsize, T<:Real} <: AbstractResult
    """
    Store the values of entanglement entropy and correlation function over multiple samples after time evolution.
    """
    type::DataType
    b::Int
    n::Real
    op::String
    nsamp::Int
    entropies::Vector{T}
    corrs::Matrix{T}
    EntrCorrResults{T}(b::Int, lsize::Int; n=1, op="Sz", nsamp::Int=100) where T<:Real = 
        new{lsize, T}(T, b, n, op, nsamp, Vector{T}(undef, nsamp), Matrix{T}(undef, lsize, nsamp))
end

function calculation_mean(lsize::Int, ttotal::Int, dent::AbstractDisentangler, res::AbstractResult; 
    cutoff::Real=1e-14, maxdim::Int=1<<(lsize ÷ 2))
    """
    Calculate the entanglement entropies over multiple samples. (non-Hermitian case)
    """
    ss = siteinds("S=1/2", lsize)
    
    for i in 1:res.nsamp 
        psi = MPS(Complex{res.type}, ss, "Up")
        timeevolve!(psi, ttotal, dent; cutoff=cutoff, maxdim=maxdim)
        @inbounds mps_results!(res, psi, i)
        psi = nothing
    end
end

function calculation_mean_multi(lsize::Int, ttotal::Int, dent::AbstractDisentangler, res::AbstractResult; 
    cutoff::Real=1e-14, maxdim::Int=1<<(lsize ÷ 2))
    """
    Calculate the entanglement entropies over multiple samples using multithreads. (non-Hermitian case)
    """

    Threads.@threads for i in 1:res.nsamp 
        ss = siteinds("S=1/2", lsize)
        psi = MPS(Complex{res.type}, ss, "Up")
        timeevolve!(psi, ttotal, dent; cutoff=cutoff, maxdim=maxdim)
        @inbounds mps_results!(res, psi, i)
        psi = nothing
    end
end

function mps_results!(res::EntropyResults, psi::MPS, i::Int)
    """
    Measure the entanglement entropy and store it in `res`.
    """
    res.entropies[i] = ent_entropy(psi, res.b, res.n)
end

function mps_results!(res::EntrCorrResults, psi::MPS, i::Int)
    """
    Measure the entanglement entropy and correlation function and store them in `res`.
    """
    res.entropies[i] = ent_entropy(psi, res.b, res.n)
    res.corrs[:, i] .= correlation_vec(psi, res.op, res.op)
end

