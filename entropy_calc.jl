using Statistics
using HDF5
include("time_evolution.jl")

function entropy_sample(lsize::Int, ttotal::Int, prob::Real, eta::Real, b::Int=lsize ÷ 2, which_ent::Real=1; 
    cutoff::Real=1e-14, ent_cutoff::Real=1e-12)
    """
    Calculate the final entanglement entropy of the MPS after time evolution. (non-Hermitian case)
    """
    ss = siteinds("S=1/2", lsize)
    psi0 = MPS(ss, "Up")
    psi = mps_evolve(psi0, ttotal, prob, eta; cutoff=cutoff)
    return Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
end

function entropy_sample(lsize::Int, ttotal::Int, prob::Real, para::Tuple{Real, Real}, b::Int=lsize ÷ 2, which_ent::Real=1; 
    cutoff::Real=1e-14, ent_cutoff::Real=1e-12)
    """
    Calculate the final entanglement entropy of the MPS after time evolution. (weak measurement case)
    """
    ss = siteinds("S=1/2", lsize)
    psi0 = MPS(ss, "Up")
    psi = mps_evolve(psi0, ttotal, prob, para; cutoff=cutoff)
    return Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
end

function entropy_mean(lsize::Int, ttotal::Int, prob::Real, eta::Real, b::Int=lsize ÷ 2, which_ent::Real=1; 
    numsamp::Int=10, cutoff::Real=1e-14, ent_cutoff::Real=1e-12, retstd::Bool=false)
    """
    Calculate the mean entanglement entropy over multiple samples.
    """
    ss = siteinds("S=1/2", lsize)
    psi0 = MPS(ss, "Up")

    entropies = Float64[]
    for _ in 1:numsamp 
        psi = mps_evolve(psi0, ttotal, prob, eta; cutoff=cutoff)
        entropy = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
        push!(entropies, entropy)
    end

    mean_entropy = mean(entropies)
    if retstd==false
        return mean_entropy
    else
        std_entropy = stdm(entropies, mean_entropy; corrected=false)
        return mean_entropy, std_entropy
    end
end

function entropy_mean(lsize::Int, ttotal::Int, prob::Real, para::Tuple{Real, Real}, b::Int=lsize ÷ 2, which_ent::Real=1; 
    numsamp::Int=10, cutoff::Real=1e-14, ent_cutoff::Real=1e-12, retstd::Bool=false)
    """
    Calculate the mean entanglement entropy over multiple samples. (weak measurement case)
    """
    ss = siteinds("S=1/2", lsize)
    psi0 = MPS(ss, "Up")

    entropies = Float64[]
    for _ in 1:numsamp 
        psi = mps_evolve(psi0, ttotal, prob, para; cutoff=cutoff)
        entropy = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
        push!(entropies, entropy)
    end

    mean_entropy = mean(entropies)
    if retstd==false
        return mean_entropy
    else
        std_entropy = stdm(entropies, mean_entropy; corrected=false)
        return mean_entropy, std_entropy
    end
end

let
    Ls = 10:10:50
    p0, η0 = 0.5, 0.5
    ps = 0.0:0.05:1.0
    ηs = 0.0:0.5:2.0
    nprob, neta = length(ps), length(ηs)

    prob_scales_mean = []
    prob_scales_std = []
    eta_scales_mean = []
    eta_scales_std = []

    for l in Ls
        tt = 4l
        b = l ÷ 2

        mean_prob = zeros(nprob)
        std_prob = zeros(nprob)
        for i in 1:nprob
            p = ps[i]
            mean_prob[i], std_prob[i] = entropy_mean(l, tt, p, η0, b; numsamp=10, retstd=true)
        end
        push!(prob_scales_mean, mean_prob)
        push!(prob_scales_std, std_prob)

        mean_eta = zeros(neta)
        std_eta = zeros(neta)
        for i in 1:neta
            η = ηs[i]
            mean_eta[i], std_eta[i] = entropy_mean(l, tt, p0, η, b; numsamp=10, retstd=true)
        end
        push!(eta_scales_mean, mean_eta)
        push!(eta_scales_std, std_eta)
    end

    prob_scales_mean = hcat(prob_scales_mean...)
    prob_scales_std = hcat(prob_scales_std...)
    eta_scales_mean = hcat(eta_scales_mean...)
    eta_scales_std = hcat(eta_scales_std...)

    h5open("entropy_scale_data.h5", "w") do file
        write(file, "ps", collect(ps))
        write(file, "ηs", collect(ηs))
        write(file, "Ls", collect(Ls))
        
        write(file, "prob_scales_mean", prob_scales_mean)
        write(file, "prob_scales_std", prob_scales_std)
        write(file, "eta_scales_mean", eta_scales_mean)
        write(file, "eta_scales_std", eta_scales_std)
    end
end