using Statistics
using Base.Threads
using ITensors, ITensorMPS
ITensors.BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)

include("time_evolution.jl")

function entropy_sample(lsize::Int, ttotal::Int, prob::Real, eta::Real, which_ent::Real=1; 
    cutoff::Real=1e-12, ent_cutoff::Real=1e-12, restype::DataType=Float64)
    """
    Calculate the final entanglement entropy of the MPS after time evolution. (non-Hermitian case)
    """
    ss = siteinds("S=1/2", lsize)
    b = lsize ÷ 2

    psi = MPS(Complex{restype}, ss, "Up")
    mps_evolve!(psi, ttotal, prob, eta; cutoff=cutoff)
    return Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
end

function entropy_sample(lsize::Int, ttotal::Int, prob::Real, para::Tuple{Real, Real}, which_ent::Real=1; 
    cutoff::Real=1e-12, ent_cutoff::Real=1e-12, restype::DataType=Float64)
    """
    Calculate the final entanglement entropy of the MPS after time evolution. (weak measurement case)
    """
    ss = siteinds("S=1/2", lsize)
    b = lsize ÷ 2

    psi = MPS(Complex{restype}, ss, "Up")
    mps_evolve!(psi, ttotal, prob, para; cutoff=cutoff)
    return Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
end

function entropy_mean(lsize::Int, ttotal::Int, prob::Real, eta::Real, which_ent::Real=1; 
    numsamp::Int=10, cutoff::Real=1e-12, ent_cutoff::Real=1e-12, retstd::Bool=false, restype::DataType=Float64)
    """
    Calculate the mean entanglement entropy over multiple samples. (non-Hermitian case)
    """
    ss = siteinds("S=1/2", lsize)
    b = lsize ÷ 2
    eta = restype(eta)
    psi0 = MPS(Complex{restype}, ss, "Up")
    # mean value of `numsamp` samples
    entropies = Vector{restype}(undef, numsamp)
    for i in 1:numsamp 
        psi = mps_evolve(psi0, ttotal, prob, eta; cutoff=cutoff)
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

function entropy_mean_multi(lsize::Int, ttotal::Int, prob::Real, eta::Real, which_ent::Real=1; 
    numsamp::Int=10, cutoff::Real=1e-12, ent_cutoff::Real=1e-12, retstd::Bool=false, restype::DataType=Float64)
    """
    Calculate the mean entanglement entropy over multiple samples. (non-Hermitian case)
    """
    # mean value of `numsamp` samples
    entropies = Vector{restype}(undef, numsamp)
    b = lsize ÷ 2
    eta = restype(eta)

    @threads for i in 1:numsamp 
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

function entropy_mean_spawn(lsize::Int, ttotal::Int, prob::Real, eta::Real, which_ent::Real=1; 
    numsamp::Int=10, cutoff::Real=1e-12, ent_cutoff::Real=1e-12, retstd::Bool=false, restype::DataType=Float64)
    """
    Calculate the mean entanglement entropy over multiple samples. (non-Hermitian case)
    """
    eta = restype(eta)
    # mean value of `numsamp` samples
    entropies = fetch.([@spawn entropy_sample(lsize, ttotal, prob, eta, which_ent; 
        cutoff=cutoff, ent_cutoff=ent_cutoff, restype=restype) for _ in 1:numsamp])
    mean_entropy = mean(entropies)
    # return std if needed
    if retstd==false
        return mean_entropy
    else
        std_entropy = stdm(entropies, mean_entropy; corrected=false)
        return mean_entropy, std_entropy
    end
end

function entropy_mean(lsize::Int, ttotal::Int, prob::Real, para::Tuple{Real, Real}, which_ent::Real=1; 
    numsamp::Int=10, cutoff::Real=1e-12, ent_cutoff::Real=1e-12, retstd::Bool=false, restype::DataType=Float64)
    """
    Calculate the mean entanglement entropy over multiple samples. (weak measurement case)
    """
    ss = siteinds("S=1/2", lsize)
    b = lsize ÷ 2
    para = restype.(para)
    psi0 = MPS(Complex{restype}, ss, "Up")

    entropies = Vector{restype}(undef, numsamp)
    for i in 1:numsamp 
        psi = mps_evolve(psi0, ttotal, prob, para; cutoff=cutoff)
        entropies[i] = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
    end

    mean_entropy = mean(entropies)
    if retstd==false
        return mean_entropy
    else
        std_entropy = stdm(entropies, mean_entropy; corrected=false)
        return mean_entropy, std_entropy
    end
end

function entropy_mean_multi(lsize::Int, ttotal::Int, prob::Real, para::Tuple{Real, Real}, which_ent::Real=1; 
    numsamp::Int=10, cutoff::Real=1e-12, ent_cutoff::Real=1e-12, retstd::Bool=false, restype::DataType=Float64)
    """
    Calculate the mean entanglement entropy over multiple samples. (weak measurement case)
    """
    entropies = Vector{restype}(undef, numsamp)
    b = lsize ÷ 2
    para = restype.(para)

    @threads for i in 1:numsamp 
        ss = siteinds("S=1/2", lsize)
        psi = MPS(Complex{restype}, ss, "Up")
        mps_evolve!(psi, ttotal, prob, para; cutoff=cutoff)
        @inbounds entropies[i] = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
        psi = nothing
        ss = nothing
    end

    mean_entropy = mean(entropies)
    if retstd==false
        return mean_entropy
    else
        std_entropy = stdm(entropies, mean_entropy; corrected=false)
        return mean_entropy, std_entropy
    end
end

function entropy_mean_spawn(lsize::Int, ttotal::Int, prob::Real, para::Tuple{Real, Real}, which_ent::Real=1; 
    numsamp::Int=10, cutoff::Real=1e-12, ent_cutoff::Real=1e-12, retstd::Bool=false, restype::DataType=Float64)
    """
    Calculate the mean entanglement entropy over multiple samples. (weak measurement case)
    """
    para = restype.(para)
    # mean value of `numsamp` samples
    entropies = fetch.([@spawn entropy_sample(lsize, ttotal, prob, para, which_ent; 
        cutoff=cutoff, ent_cutoff=ent_cutoff, restype=restype) for _ in 1:numsamp])
    mean_entropy = mean(entropies)
    # return std if needed
    if retstd==false
        return mean_entropy
    else
        std_entropy = stdm(entropies, mean_entropy; corrected=false)
        return mean_entropy, std_entropy
    end
end


