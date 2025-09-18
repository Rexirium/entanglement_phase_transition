using Statistics
using Base.Threads
include("time_evolution.jl")
ITensors.BLAS.set_num_threads(1)

function entropy_sample(lsize::Int, ttotal::Int, prob::Real, eta::Real, b::Int=lsize ÷ 2, which_ent::Real=1; 
    cutoff::Real=1e-12, ent_cutoff::Real=1e-10)
    """
    Calculate the final entanglement entropy of the MPS after time evolution. (non-Hermitian case)
    """
    ss = siteinds("S=1/2", lsize)
    psi0 = MPS(ss, "Up")
    psi = mps_evolve(psi0, ttotal, prob, eta; cutoff=cutoff)
    return Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
end

function entropy_sample(lsize::Int, ttotal::Int, prob::Real, para::Tuple{Real, Real}, b::Int=lsize ÷ 2, which_ent::Real=1; 
    cutoff::Real=1e-12, ent_cutoff::Real=1e-10)
    """
    Calculate the final entanglement entropy of the MPS after time evolution. (weak measurement case)
    """
    ss = siteinds("S=1/2", lsize)
    psi0 = MPS(ss, "Up")
    psi = mps_evolve(psi0, ttotal, prob, para; cutoff=cutoff)
    return Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
end

function entropy_mean(lsize::Int, ttotal::Int, prob::Real, eta::Real, b::Int=lsize ÷ 2, which_ent::Real=1; 
    numsamp::Int=10, cutoff::Real=1e-12, ent_cutoff::Real=1e-10, retstd::Bool=false)
    """
    Calculate the mean entanglement entropy over multiple samples. (non-Hermitian case)
    """
    ss = siteinds("S=1/2", lsize)
    psi0 = MPS(ss, "Up")
    # mean value of `numsamp` samples
    entropies = zeros(Float64, numsamp)
    for i in 1:numsamp 
        psi = mps_evolve(psi0, ttotal, prob, eta; cutoff=cutoff)
        entropy = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
        entropies[i] = entropy
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
    numsamp::Int=10, cutoff::Real=1e-12, ent_cutoff::Real=1e-10, retstd::Bool=false)
    """
    Calculate the mean entanglement entropy over multiple samples. (non-Hermitian case)
    """
    # mean value of `numsamp` samples
    entropies = zeros(Float64, numsamp)
    mylock = ReentrantLock()
    @threads for i in 1:numsamp 
        ss = siteinds("S=1/2", lsize)
        psi = MPS(ss, "Up")
        mps_evolve!(psi, ttotal, prob, eta; cutoff=cutoff)
        entropy = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
        @lock mylock entropies[i] = entropy
        # println("entropy = $entropy, thread $(threadid())")  # check for multithread
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
    numsamp::Int=10, cutoff::Real=1e-12, ent_cutoff::Real=1e-10, retstd::Bool=false)
    """
    Calculate the mean entanglement entropy over multiple samples. (weak measurement case)
    """
    ss = siteinds("S=1/2", lsize)
    psi0 = MPS(ss, "Up")

    entropies = zeros(Float64, numsamp)
    for i in 1:numsamp 
        psi = mps_evolve(psi0, ttotal, prob, para; cutoff=cutoff)
        entropy = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
        entropies[i] = entropy
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
    numsamp::Int=10, cutoff::Real=1e-12, ent_cutoff::Real=1e-10, retstd::Bool=false)
    """
    Calculate the mean entanglement entropy over multiple samples. (weak measurement case)
    """
    entropies = zeros(Float64, numsamp)
    mylock = ReentrantLock()

    @threads for i in 1:numsamp 
        ss = siteinds("S=1/2", lsize)
        psi = MPS(ss, "Up")
        mps_evolve!(psi, ttotal, prob, para; cutoff=cutoff)
        entropy = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
        @lock mylock entropies[i] = entropy
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
    # example usage
    L = 10
    T, b = 4L, L ÷ 2
    prob = 0.1
    eta = 1.0
    numsamp = 20

    mean_entropy, std_entropy = entropy_mean_multi(L, T, prob, eta; numsamp=numsamp, retstd=true)
    println("Mean entropy (non-Hermitian): $mean_entropy ± $std_entropy")

    para = (0.5, 0.5)
    mean_entropy, std_entropy = entropy_mean_multi(L, T, prob, para; numsamp=numsamp, retstd=true)
    println("Mean entropy (weak measurement): $mean_entropy ± $std_entropy")
end
