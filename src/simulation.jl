using Statistics
using ITensors, ITensorMPS
ITensors.BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)

include("time_evolution.jl")

struct CalcResult{T}
    mean_entropy::T
    std_entropy::T
    mean_corrs::Vector{T}
    std_corrs::Vector{T}
end

function entropy_sample(lsize::Int, ttotal::Int, prob::Real, eta::Real, which_ent::Real=1; 
    cutoff::Real=1e-12, ent_cutoff::Real=1e-12, restype::DataType=Float64) where Tp<:Real
    """
    Calculate the final entanglement entropy of the MPS after time evolution. 
    """
    b = lsize ÷ 2
    eta = restype(eta)

    ss = siteinds("S=1/2", lsize)
    psi = MPS(Complex{restype}, ss, "Up")
    mps_evolve!(psi, ttotal, prob, eta; cutoff=cutoff)

    entropy = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
    return entropy
end

function calculation_sample(lsize::Int, ttotal::Int, prob::Real, eta::Real, which_ent::Real=1, which_op::String="Sz"; 
    cutoff::Real=1e-12, ent_cutoff::Real=1e-12, restype::DataType=Float64)
    """
    Calculate the final entanglement entropy and correlation function of the MPS after time evolution. 
    """
    b = lsize ÷ 2
    eta = restype(eta)

    ss = siteinds("S=1/2", lsize)
    psi = MPS(Complex{restype}, ss, "Up")
    mps_evolve!(psi, ttotal, prob, eta; cutoff=cutoff)

    entropy = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
    corrs = correlation_vec(psi, which_op, which_op)
    return entropy, corrs
end

function entropy_mean(lsize::Int, ttotal::Int, prob::Real, eta::Real, which_ent::Real=1; 
    numsamp::Int=10, cutoff::Real=1e-12, ent_cutoff::Real=1e-12, retstd::Bool=false, restype::DataType=Float64)
    """
    Calculate the mean entanglement entropy over multiple samples. (non-Hermitian case)
    """
    b = lsize ÷ 2
    eta = restype(eta)
    ss = siteinds("S=1/2", lsize)
    # mean value of `numsamp` samples
    entropies = Vector{restype}(undef, numsamp)
    for i in 1:numsamp 
        psi = MPS(Complex{restype}, ss, "Up")
        mps_evolve!(psi, ttotal, prob, eta; cutoff=cutoff)
        entropies[i] = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
    end
    mean_entropy = mean(entropies)
    # return std if needed
    if retstd==false
        return mean_entropy
    else
        std_entropy = stdm(entropies, mean_entropy; corrected=false)
        return mean_entropy, std_entropy
    end
end

function calculation_mean(lsize::Int, ttotal::Int, prob::Real, eta::Real, which_ent::Real=1, which_op::String="Sz"; 
    numsamp::Int=10, cutoff::Real=1e-12, ent_cutoff::Real=1e-12, retstd::Bool=false, restype::DataType=Float64)
    """
    Calculate the mean entanglement entropy over multiple samples. (non-Hermitian case)
    """
    b = lsize ÷ 2
    eta = restype(eta)
    ss = siteinds("S=1/2", lsize)
    # mean value of `numsamp` samples
    entropies = Vector{restype}(undef, numsamp)
    corrs = Matrix{restype}(undef, lsize, numsamp)
    for i in 1:numsamp 
        psi = MPS(Complex{restype}, ss, "Up")
        mps_evolve!(psi, ttotal, prob, eta; cutoff=cutoff)
        entropies[i] = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
        corrs[:, i] .= correlation_vec(psi, which_op, which_op)  
    end
    mean_entropy = mean(entropies)
    mean_corrs = vec(mean(corrs, dims=2))
    # return std if needed
    if retstd==false
        return mean_entropy, mean_corrs
    else
        std_entropy = stdm(entropies, mean_entropy; corrected=false)
        std_corrs = vec(stdm(corrs, mean_corrs; dims=2, corrected=false))
        return CalcResult{restype}(mean_entropy, std_entropy, mean_corrs, std_corrs)
    end
end

function entropy_mean_multi(lsize::Int, ttotal::Int, prob::Real, eta::Real, which_ent::Real=1; 
    numsamp::Int=10, cutoff::Real=1e-12, ent_cutoff::Real=1e-12, retstd::Bool=false, restype::DataType=Float64)
    """
    Calculate the mean entanglement entropy over multiple samples. (non-Hermitian case)
    """
    # mean value of `numsamp` samples
    b = lsize ÷ 2
    eta = restype(eta)

    entropies = Vector{restype}(undef, numsamp)
    Threads.@threads for i in 1:numsamp 
        ss = siteinds("S=1/2", lsize)
        psi = MPS(Complex{restype}, ss, "Up")
        mps_evolve!(psi, ttotal, prob, eta; cutoff=cutoff)
        @inbounds entropies[i] = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
        psi = nothing
        ss = nothing
    end
    mean_entropy = mean(entropies)
    # return std if needed
    if retstd==false
        return mean_entropy
    else
        std_entropy = stdm(entropies, mean_entropy; corrected=false)
        return mean_entropy, std_entropy
    end
end

function calculation_mean_multi(lsize::Int, ttotal::Int, prob::Real, eta::Real, which_ent::Real=1, which_op::String="Sz"; 
    numsamp::Int=10, cutoff::Real=1e-12, ent_cutoff::Real=1e-12, retstd::Bool=false, restype::DataType=Float64)
    """
    Calculate the mean entanglement entropy over multiple samples. (non-Hermitian case)
    """
    # mean value of `numsamp` samples
    b = lsize ÷ 2
    eta = restype(eta)

    entropies = Vector{restype}(undef, numsamp)
    corrs = Matrix{restype}(undef, lsize, numsamp)
    Threads.@threads for i in 1:numsamp 
        ss = siteinds("S=1/2", lsize)
        psi = MPS(Complex{restype}, ss, "Up")
        mps_evolve!(psi, ttotal, prob, eta; cutoff=cutoff)
        @inbounds entropies[i] = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
        @inbounds corrs[:, i] .= correlation_vec(psi, which_op, which_op)
        psi = nothing
        ss = nothing
    end
    mean_entropy = mean(entropies)
    mean_corrs = vec(mean(corrs, dims=2))
    # return std if needed
    if retstd==false
        return mean_entropy, mean_corrs
    else
        std_entropy = stdm(entropies, mean_entropy; corrected=false)
        std_corrs = vec(stdm(corrs, mean_corrs; dims=2, corrected=false))
        return CalcResult{restype}(mean_entropy, std_entropy, mean_corrs, std_corrs)
    end
end


