function linregress(xs, ys)
    n = length(xs)
    A = [ones(n) xs]
    coeffs = A \ ys
    return coeffs[2]
end

function linregress(xs, ys, yerrs)
    n = length(xs)
    ws = normalize(1 ./(yerrs .^ 2), 1)
    W = diagm(sqrt.(ws))
    A = [ones(n) xs]
    Aw = W * A
    yw = W * ys
    coeffs = Aw \ yw
    return coeffs[2]
end