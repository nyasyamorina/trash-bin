include("wavlet - funcs.jl")
using Polynomials


function genDaubechiesCoefs(N::Integer)
    (N < 1) && throw(ArgumentError("N-order Daubechies wavelet needs `N` > 0"))
    poly = Polynomial([1, 1])^N
    for q ∈ roots(Polynomial([binomial(N - 1 + k, k) for k ∈ 0:N - 1]))
        r = 1 - 2 * (q + √(q * (q - 1)))
        (abs2(r) ≥ 1) && (r = 1 / r)        # ?
        poly *= Polynomial([-r, 1])
    end
    c = reverse(real.(poly.coeffs))
    return CoefP((c .* (2 / sum(c)))...)
end


# doesn't need reversing version

function genDaubechiesCoefs_(N::Integer)
    (N < 1) && throw(ArgumentError("N-order Daubechies wavelet needs `N` > 0"))
    poly = Polynomial([1, 1])^N
    for q ∈ roots(Polynomial([binomial(N - 1 + k, k) for k ∈ 0:N - 1]))
        r = 1 - 2 * (q - √(q * (q - 1)))
        (abs2(r) ≤ 1) && (r = 1 / r)        # ?
        poly *= Polynomial([-r, 1])
    end
    c = real.(poly.coeffs)
    return CoefP((c .* (2 / sum(c)))...)
end
