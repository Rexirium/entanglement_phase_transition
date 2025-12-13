using MKL
using LinearAlgebra, NDTensors
using Random
using Plots, LaTeXStrings

X = [0 1; 1 0]
Y = [0 -im; im 0]
Z = [1 0; 0 -1]

pauli = [X, Y, Z]

function sin_sampler(rng::AbstractRNG=Random.default_rng())
    xi = rand(rng)
    if rand(rng) < sin(π * xi)
        return π * xi
    end
    return sin_sampler(rng)
end

function sin_sampler(n::Int, rng::AbstractRNG=Random.default_rng(); max_attempts=100*n)
    samples = Float64[]
    accepted = 0
    attempts = 0
    while accepted < n && attempts < max_attempts
        xi = rand(rng)
        if rand(rng) < sin(π * xi)
            push!(samples, π * xi)
            accepted += 1
        end
        attempts += 1
    end
    return samples
end

function random_unitary(d::Int, rng::AbstractRNG=Random.default_rng())
    M = randn(rng, ComplexF64, d, d)
    U, _ = NDTensors.qr_positive(M)
    return U
end

function random_unitary_origin(d::Int, rng::AbstractRNG=Random.default_rng())
    χ, φ = 2π * rand(rng), 2π * rand(rng)
    θ = sin_sampler(rng)
    phasep = exp(im * (χ+φ)/2) * cos(θ/2)
    phasem = exp(im * (χ-φ)/2) * sin(θ/2)
    return [phasep' (-phasem); 
            phasem' phasep]
end

let 
    N = 2000
    d = 2
    rng = MersenneTwister(1234)
    ts = range(0, 2π, length=200)
    xs, ys = cos.(ts), sin.(ts)
    zs = zeros(200)

    psi0 = [1, 0]
    rho0 = kron(psi0, psi0')
    points = zeros(Float64, N, 3)
    for i in 1:N
        U = random_unitary(d)
        rho = U * rho0 * U'
        for j in 1:3
            points[i, j] = real(tr(pauli[j] * rho))
        end
    end

    scatter(points[:, 1], points[:, 2], points[:, 3], markersize=2,
        xlabel=L"x", ylabel=L"y", zlabel=L"z", size=(800,600))
    plot!(sphere=(0, 1), alpha=0.2, legend=false)
    plot!(xs, ys, zs, lw=2, lc=:red, leg=false)
end
