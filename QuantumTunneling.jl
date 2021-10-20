"""
    EigenFunc(m, U₀, a, E)

Return eigenfunction of the finite barrier potential corresponds to eigenvalue E.

Note that barrier potential in range [0,a].

# Arguments
- `U::Real`: height of barrier, must be larger than 0.
- `a::Real`: width of barrier, must be larger than 0.
- `m::Real`: mass of particle, must be larger than 0.
- `E::Real`: energy of particle, must be larger than 0.
"""
function EigenFunc(U::Real, a::Real, m::Real, E::Real)
    k1 = √(2m * E) / ħ

    if E ≈ U
        t = a * k1
        A2 = t * (t - 2im) / (4 + t^2 * ħ^-4)
        C1 = (exp(-im * t) * (4 + 2im * t * ħ^-2)) / (4 + t^2)
        b1 = 2k1 * ComplexF64(t, 2)
        b2 = (4 - 2t * ComplexF64(t, 3))
        b1, b2 = [b1 b2] ./ (4 - 4im * t - t^2)

        return x -> begin
            if x < 0
                return exp(im * k1 * x) + A2 * exp(-im * k1 * x)
            elseif x < a
                return b1 * x + b2
            else
                return C1 * exp(im * k1 * x)
            end
        end

    else
        k2 = √ComplexF64(2m * (E - U)) / ħ

        t = inv(exp(im * k2 * a) * (k1 - k2)^2 - exp(-im * k2 * a) * (k1 + k2)^2)
        A2 = 2im * sin(k2 * a) * (k1^2 - k2^2) * t
        B1 = -2exp(-im * k2 * a) * k1 * (k1 + k2) * t
        B2 =  2exp( im * k2 * a) * k1 * (k1 - k2) * t
        C1 = -4exp(-im * k1 * a) * k1 * k2 * t

        return x -> begin
            if x < 0
                return exp(im * k1 * x) + A2 * exp(-im * k1 * x)
            elseif x < a
                return B1 * exp(im * k2 * x) + B2 * exp(-im * k2 * x)
            else
                return C1 * exp(im * k1 * x)
            end
        end

    end
end


function barrier(U::Real, a::Real)
    return x -> (0 < x < a) ? U : 0
end


using Plots, PlotThemes, Printf
theme(:juno)

const ħ = 1.


function main()
    fps = 24
    time = 30

    anim = @animate for E ∈ range(eps(), 2., length = time * fps)
        ψ = EigenFunc(1, 1, 1, E)

        fig = plot([0., 0., 1., 1., 0.], [1., -1., -1., 1., 1.], label="barrier")
        plot!(fig, [real imag].∘((x->0.6x) ∘ ψ), -5., 5., framestyle = :zerolines,
            label=["Re" "Im"], color=[RGB(.4,.6,1.) RGB(1.,.3,.1)], lw=2)
        ylims!(fig, (-2., 2.))
        title!(fig, "E=" * (@sprintf "%.2f" E) * "U₀")

        fig
    end

    mp4(anim, "out.mp4")
end

main();
