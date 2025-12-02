using ITensors, ITensorMPS

function correlation(psi::MPS, ops1::String, ops2::String, i::Int, j::Int)
    idxs = siteinds(psi)[[i, j]]
    op1 = op(ops1, idxs[1])
    op2 = op(ops2, idxs[2])
    start, stop = sort((i, j))
    (start ≤ 0 || stop > length(psi)) && error("The sites do not exist!")
    orthogonalize!(psi, start)

    C = psi[start] * op1
    if start == stop
        Cdag = dag(psi[start] * op2)
        C *= Cdag
        return real(scalar(C))
    end
    ir = linkind(psi, start)
    C *= dag(prime(prime(psi[start], tags="Site"), ir))
    for j in (start+1):(stop-1)
        C *= psi[j]
        C *= dag(prime(psi[j], tags="Link"))
    end
    C *= psi[stop] * op2
    il = linkind(psi, stop-1)
    C *= dag(prime(prime(psi[stop], tags="Site"), il))
    return real(scalar(C))
end

function correlation(psi::MPS, ops1::String, ops2::String, dist::Int)
    Lsize = length(psi)
    dist > Lsize - 1 && error("The distance is too large!")
    start = 1 + (Lsize - dist - 1) ÷ 2
    stop = start + dist
    return correlation(psi, ops1, ops2, start, stop)
end

function correlation_vec(psi::MPS, ops1::String, ops2::String)
    Lsize = length(psi)
    corrs = zeros(Float64, Lsize)
    for dist in 0:(Lsize - 1)
        corrs[dist+1] = correlation(psi, ops1, ops2, dist)
    end
    return corrs
end

let 
    ss = siteinds("S=1/2", 10)
    psi = random_mps(ss; linkdims=4)
    @time println(correlation_vec(psi, "Sx", "Sx"))
    @time println(correlation_matrix(psi, "Sx", "Sx")[4, 4])
end