include("time_evolution.jl")
ITensors.BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)

abstract type AbstractResult end

mutable struct EntropySample{T} <: AbstractResult
    """
    Store the entanglement entropy result after time evolution.
    """
    type::DataType
    b::Int
    n::Real
    entropy::T
    EntropySample{T}(b::Int, n = 1) where T<:Real = new{T}(T, b, n, zero(T))
end

mutable struct EntrCorrSample{T} <: AbstractResult
    """
    Store the entanglement entropy and correlation function results after time evolution.
    """
    type::DataType
    b::Int
    len::Int
    n::Real
    op::String
    entropy::T
    corrs::Vector{T}
    CalculationSample{T}(len::Int, n = 1, op = "Sz") where T<:Real = new{T}(T, len÷2, len, n, op, zero(T), zeros(T, len))
end

mutable struct EntropyResults{T} <: AbstractResult
    """
    Store the mean and std of entanglement entropy over multiple samples after time evolution.
    """
    type::DataType
    b::Int
    n::Real
    nsamp::Int
    entropies::Vector{T}
    EntropyResults{T}(b::Int, n=1; nsamp::Int = 100) where T<:Real = new{T}(T, b, n, nsamp, zeros(T, nsamp))
end

mutable struct EntrCorrResults{T} <: AbstractResult
    """
    Store the mean and std of entanglement entropy and correlation function over multiple samples after time evolution.
    """
    type::DataType
    b::Int
    len::Int
    n::Real
    op::String
    nsamp::Int
    entropies::Vector{T}
    corrs::Matrix{T}
    EntrCorrResults{T}(len::Int, n=1, op="Sz"; nsamp::Int=100) where T<:Real = 
        new{T}(T, len÷2, len, n, op, nsamp, zeros(T, nsamp), zeros(T, len, nsamp))
end

function calculation_sample(lsize::Int, ttotal::Int, prob::Real, eta::Real, res::AbstractResult; cutoff::Real=1e-12)
    """
    Calculate the final properties of the MPS after time evolution. 
    """
    eta = (res.type)(eta)

    ss = siteinds("S=1/2", lsize)
    psi = MPS(Complex{res.type}, ss, "Up")
    mps_evolve!(psi, ttotal, prob, eta; cutoff=cutoff)

    mps_measure!(res, psi)
end

function mps_measure!(res::EntropySample{T}, psi::MPS) where T<:Real
    """
    Measure the entanglement entropy and store it in `res`.
    """
    res.entropy = ent_entropy(psi, res.b, res.n)
end

function calculation_mean(lsize::Int, ttotal::Int, prob::Real, eta::Real, res::AbstractResult; cutoff::Real=1e-12)
    """
    Calculate the mean entanglement entropy over multiple samples. (non-Hermitian case)
    """
    eta = (res.type)(eta)
    ss = siteinds("S=1/2", lsize)
    
    for i in 1:res.nsamp 
        psi = MPS(Complex{res.type}, ss, "Up")
        mps_evolve!(psi, ttotal, prob, eta; cutoff=cutoff)
        mps_measure!(res, psi, i)
    end
end

function calculation_mean_multi(lsize::Int, ttotal::Int, prob::Real, eta::Real, res::AbstractResult; cutoff::Real=1e-12)
    """
    Calculate the mean entanglement entropy over multiple samples using multithreads. (non-Hermitian case)
    """
    eta = (res.type)(eta)

    Threads.@threads for i in 1:res.nsamp 
        ss = siteinds("S=1/2", lsize)
        psi = MPS(Complex{res.type}, ss, "Up")
        mps_evolve!(psi, ttotal, prob, eta; cutoff=cutoff)
        @inbounds mps_measure!(res, psi, i)
        psi = nothing
        ss = nothing
    end
end

function mps_measure!(res::EntropyResults{T}, psi::MPS, i::Int) where T<:Real
    """
    Measure the entanglement entropy and store it in `res`.
    """
    res.entropies[i] = ent_entropy(psi, res.b, res.n)
end

function mps_measure!(res::EntrCorrResults{T}, psi::MPS, i::Int) where T<:Real
    """
    Measure the entanglement entropy and correlation function and store them in `res`.
    """
    res.entropies[i] = ent_entropy(psi, res.b, res.n)
    res.corrs[:, i] .= correlation_vec(psi, res.op, res.op)
end

