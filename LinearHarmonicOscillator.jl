φ(x) = (0.5pi)^-0.25 * exp((5im - x) * x)   # wave func at t=0
const N = 40        # numbers of wave func to simulate

const m = 1.        # mass
const ω = 1.        # intensity
const ħ = 1.        # Planck constant

x = -10 : 0.05 : 10     # drawing range
const time = 4pi                # total time of particle

const speed = 0.5               # play speed of video
const fps = 60                  # fps of video (Integer)
outputfile = "D:/Jl/anim.mp4"   # path of video




println("Initializing...")
using QuadGK, Plots, Printf

try
    using PlotThemes
    theme(:juno)
catch _
finally
    Plots.scalefontsizes(2.8)
end


function HermiteFunction(coef::Vector{Float64}, α::Float64)
    return x -> begin
        result = zero(Float64)
        for a ∈ coef
            result = result * α * x + a
        end
        return result * exp(-α^2 / 2 * x * x)
    end
end


struct LinearHarmonicOscillator
    m::Float64              # mass of particle
    ω::Float64              # intensity of potential field
    E::Vector{Float64}      # energy of steady state
    ψ::Vector{Function}     # steady state wave function
    N::UInt                 # count of E end ψ

    function LinearHarmonicOscillator(m::Float64, ω::Float64, N::Integer)
        E = Vector{Float64}(undef, 0)
        ψ = Vector{Function}(undef, 0)

        coef = [pi^-0.25,]
        for n ∈ 0:N-1
            push!(E, ħ * ω * (n + 0.5))
            push!(ψ, HermiteFunction(coef, sqrt(m * ω / ħ)))

            if n < N-1
                nextcoef = Vector{Float64}(undef, n+2)
                nextcoef[1:n+1] .= 2 .* coef
                nextcoef[n+2] = 0.
                nextcoef[3:n+2] .-= collect(n:-1:1) .* coef[1:n]
                coef = (1 / sqrt(2n + 2)) .* nextcoef
            end
        end

        new(m, ω, E, ψ, N)
    end
end


function potential(lho::LinearHarmonicOscillator)
    return x -> 0.5 * lho.m * lho.ω^2 * x * x
end


function convert(φ::Function, lho::LinearHarmonicOscillator)
    getc(ψ) = quadgk(x -> φ(x) * conj(ψ(x)), -Inf, Inf)[1]
    c = getc.(lho.ψ)
    #png(plot(0:lho:N-1 abs2.(c)), "energy_distribution.png")
    return (x, t) -> begin
        result = zero(ComplexF64)
        for i ∈ 1:lho.N
            result +=
                c[i] *
                exp(lho.E[i] * t / ComplexF64(0., ħ)) *
                lho.ψ[i](x)
        end
        return result
    end
end


function show(Ψ::Function, x::AbstractArray, t::AbstractFloat; U=nothing)
    fig = plot(layout = @layout[a; b], framestyle = :zerolines, size = (1920, 1080))
    title!(fig[1], "time=" * (@sprintf "%.2f" t))
    xticks!(fig[1], 1:0); ylims!(fig[1], (-1., 1.)); yticks!(fig[1], [0.])
    xlabel!(fig[2], "x"); ylims!(fig[2], (0., 1.)); yticks!(fig[2], [0., 1.])

    f = Ψ.(x, t)
    plot!(fig[1],
          x, [real.(f), imag.(f)];
          label=["Re" "Im"], color=[RGB(.4, .6, 1.) RGB(1., .3, .1)], lw=2)
    plot!(fig[2],
          x, abs2.(f);
          label="Pr", color=RGB(.8, .4, 1.), lw=2)
    if U != nothing
        plot!(fig[2], x, U; label="U", lw=.7)
    end

    return fig
end


println("Converting wave function...")
lho = LinearHarmonicOscillator(m, ω, N)
U = potential(lho).(x)
U ./= max(U...)   # make sure U can be shown in figure
Ψ = convert(φ, lho)

println("Rendering frames...")
anim = @animate for t ∈ 0 : speed/fps : time
    show(Ψ, x, t; U=U)
end

println(" -> animate frames storage in ", anim.dir)


mp4(anim, outputfile)
#println("Compiling video...")
#outputfile = abspath(expanduser(outputfile))
#ffmpeg_common = "ffmpeg -v 16 -framerate $fps -i $(anim.dir)/%06d.png -crf 0 -y $(outputfile)"
#run(ffmpeg_common);
#
#println(" -> saved video on", outputfile)
