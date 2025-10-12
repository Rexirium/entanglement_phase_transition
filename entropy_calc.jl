using Statistics
using Base.Threads
include("time_evolution.jl")
ITensors.BLAS.set_num_threads(1)

function entropy_sample(lsize::Int, ttotal::Int, prob::Real, eta::Real, b::Int=lsize ÷ 2, which_ent::Real=1; 
    cutoff::Real=1e-10, ent_cutoff::Real=1e-8)
    """
    Calculate the final entanglement entropy of the MPS after time evolution. (non-Hermitian case)
    """
    ss = siteinds("S=1/2", lsize)
    psi0 = MPS(ComplexF64, ss, "Up")
    psi = mps_evolve(psi0, ttotal, prob, eta; cutoff=cutoff)
    return Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
end

function entropy_sample(lsize::Int, ttotal::Int, prob::Real, para::Tuple{Real, Real}, b::Int=lsize ÷ 2, which_ent::Real=1; 
    cutoff::Real=1e-10, ent_cutoff::Real=1e-8)
    """
    Calculate the final entanglement entropy of the MPS after time evolution. (weak measurement case)
    """
    ss = siteinds("S=1/2", lsize)
    psi0 = MPS(ComplexF64, ss, "Up")
    psi = mps_evolve(psi0, ttotal, prob, para; cutoff=cutoff)
    return Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
end

function entropy_mean(lsize::Int, ttotal::Int, prob::Real, eta::Real, b::Int=lsize ÷ 2, which_ent::Real=1; 
    numsamp::Int=10, cutoff::Real=1e-10, ent_cutoff::Real=1e-8, retstd::Bool=false)
    """
    Calculate the mean entanglement entropy over multiple samples. (non-Hermitian case)
    """
    ss = siteinds("S=1/2", lsize)
    psi0 = MPS(ComplexF64, ss, "Up")
    # mean value of `numsamp` samples
    entropies = zeros(Float64, numsamp)
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

function entropy_mean_multi(lsize::Int, ttotal::Int, prob::Real, eta::Real, b::Int=lsize ÷ 2, which_ent::Real=1; 
    numsamp::Int=10, cutoff::Real=1e-10, ent_cutoff::Real=1e-8, retstd::Bool=false)
    """
    Calculate the mean entanglement entropy over multiple samples. (non-Hermitian case)
    """
    # mean value of `numsamp` samples
    entropies = zeros(Float64, numsamp)

    @threads for i in 1:numsamp 
        ss = siteinds("S=1/2", lsize)
        psi = MPS(ComplexF64, ss, "Up")
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

function entropy_mean(lsize::Int, ttotal::Int, prob::Real, para::Tuple{Real, Real}, b::Int=lsize ÷ 2, which_ent::Real=1; 
    numsamp::Int=10, cutoff::Real=1e-10, ent_cutoff::Real=1e-8, retstd::Bool=false)
    """
    Calculate the mean entanglement entropy over multiple samples. (weak measurement case)
    """
    ss = siteinds("S=1/2", lsize)
    psi0 = MPS(ComplexF64, ss, "Up")

    entropies = zeros(Float64, numsamp)
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

function entropy_mean_multi(lsize::Int, ttotal::Int, prob::Real, para::Tuple{Real, Real}, b::Int=lsize ÷ 2, which_ent::Real=1; 
    numsamp::Int=10, cutoff::Real=1e-10, ent_cutoff::Real=1e-8, retstd::Bool=false)
    """
    Calculate the mean entanglement entropy over multiple samples. (weak measurement case)
    """
    entropies = zeros(Float64, numsamp)

    @threads for i in 1:numsamp 
        ss = siteinds("S=1/2", lsize)
        psi = MPS(ComplexF64, ss, "Up")
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

if abspath(PROGRAM_FILE) == @__FILE__
# let
    # example usage
    L = 12
    T, b = 4L, L ÷ 2
    prob = 0.5
    eta = 0.5
    numsamp = 10

    @timev  entropy_mean(L, T, prob, eta; numsamp=numsamp, retstd=true)
    @timev  entropy_mean_multi(L, T, prob, eta; numsamp=numsamp, retstd=true)
end
