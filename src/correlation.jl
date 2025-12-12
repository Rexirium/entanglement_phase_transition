using ITensors, ITensorMPS

function correlation(psi::MPS, ops1::String, ops2::String, i::Int, j::Int)
    left, right = sort((i, j))
    idxs = siteinds(psi)[[left, right]]
    op1 = op(ops1, idxs[1])
    op2 = op(ops2, idxs[2])
    (left ≤ 0 || right > length(psi)) && error("The sites do not exist!")
    orthogonalize!(psi, left)

    C = psi[left] * op1
    if left == right
        Cdag = dag(psi[left] * op2)
        C *= Cdag
        return real(scalar(C))
    end
    ir = linkind(psi, left)
    C *= dag(prime(prime(psi[left], tags="Site"), ir))
    for n in (left+1):(right-1)
        C *= psi[n]
        C *= dag(prime(psi[n], tags="Link"))
    end
    C *= psi[right] * op2
    il = linkind(psi, right-1)
    C *= dag(prime(prime(psi[right], tags="Site"), il))
    return real(scalar(C))
end

function correlation(psi::MPS, ops1::String, ops2::String, dist::Int)
    lsize = length(psi)
    dist > lsize - 1 && error("The distance is too large!")
    left = 1 + (lsize - dist - 1) ÷ 2
    right = left + dist
    return correlation(psi, ops1, ops2, left, right)
end

function correlation_vec(psi::MPS, ops1::String, ops2::String)
    lsize = length(psi)
    corrs = zeros(Float64, lsize)
    for dist in 0:(lsize - 1)
        left = 1 + (lsize - dist - 1) ÷ 2
        right = left + dist
        corrs[dist+1] = correlation(psi, ops1, ops2, left, right)
    end
    return corrs
end
#=
let 
    ss = siteinds("S=1/2", 10)
    psi = randomMPS(ss; linkdims=4)
    @time println(correlation_vec(psi, "Sx", "Sx"))
    @time println(correlation_matrix(psi, "Sx", "Sx")[4, 4])
end
=#