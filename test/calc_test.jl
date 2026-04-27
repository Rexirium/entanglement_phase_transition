using MKL
using Statistics
using ITensors, ITensorMPS
ITensors.BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)

if !isdefined(Main, :RandomUnitary)
    include("../src/RandomUnitary.jl")
    using .RandomUnitary
end

let 
    L = 12
    ss = siteinds("S=1/2", L)
    psi = randomMPS(ComplexF64, ss; linkdims=8)

    correlation_site(psi, "Z", "Z")
end