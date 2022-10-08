using Plots
colors = theme_palette(:auto)


# use `Tuple{Number, Number}` for space-time coordinates
L((x⁰,x¹), β) = (x⁰-β*x¹, x¹-β*x⁰) ./ sqrt(1-β*β)   # Lorentz transformation
sw((x⁰,x¹)) = (sqrt(x⁰*x⁰-x¹*x¹), atanh(x¹/x⁰))     # convert (x⁰,x¹) to (s,w)
R((s,w), w₀) = (s, w-w₀)                            # "Rotation"
x((s,w)) = (s*cosh(w), s*sinh(w))                   # convert (s,w) to (x⁰,x¹)

x⁰(wl) = [x⁰ for (x⁰,x¹) ∈ wl]
x¹(wl) = [x¹ for (x⁰,x¹) ∈ wl]


# generate brother's worldline
function gen_wl_brother(dτ)
    wl = NTuple{2, Float64}[]
    for τ ∈ 0:dτ:2π
        L_wl!(dτ * cos(τ), wl)
        push!(wl, (0., 0.))
        shift!((dτ, 0), wl)
    end
    shift!(wl[1], wl)
    return wl
end

# plot sister's & brother's worldline
function plot_wl(lims, scatter_pre_n, wl_sister, wl_brother; kwargs...)
    x⁰_brother = x⁰(wl_brother)
    x¹_brother = x¹(wl_brother)
    x⁰_sister = x⁰(wl_sister)
    x¹_sister = x¹(wl_sister)

    plot([-100, 100, -100, 100], [100, -100, -100, 100]; color = :black)

    plot!(x¹_sister, x⁰_sister; color = colors[1])
    plot!(x¹_brother, x⁰_brother; color = colors[3])
    plot!(x¹_sister[1:scatter_pre_n:end], x⁰_sister[1:scatter_pre_n:end]; color = colors[1], seriestype = :scatter)
    plot!(x¹_brother[1:scatter_pre_n:end], x⁰_brother[1:scatter_pre_n:end]; color = colors[3], seriestype = :scatter)
    xlims!(lims); ylims!(lims; kwargs...)
end

# Lorentz transformation for multiple worldline
function L_wl!(β, wls...)
    γ = 1 / sqrt(1 - β * β)
    for wl ∈ wls
        for idx ∈ 1:length(wl)
            x⁰, x¹ = wl[idx]
            wl[idx] = (γ * (x⁰ - β * x¹), γ * (x¹ - β * x⁰))
        end
    end
end

# shift function for multiple worldline
function shift!((dx⁰, dx¹), wls...)
    for wl ∈ wls
        for idx ∈ 1:length(wl)
            x⁰, x¹ = wl[idx]
            wl[idx] = (x⁰ - dx⁰, x¹ - dx¹)
        end
    end
end


function main()
    dτ = 0.01

    wl_brother_origin = gen_wl_brother(dτ)
    wl_sister_origin = [(τ, 0.) for τ ∈ 0:dτ:wl_brother_origin[end][1]]

    wl_sister = copy(wl_sister_origin)
    wl_brother = copy(wl_brother_origin)
    anim_sister = @animate for idx ∈ 2:length(wl_sister)
        x⁰, x¹ = wl_sister[idx]
        L_wl!(x¹ / x⁰, wl_sister, wl_brother)
        shift!(wl_sister[idx], wl_sister, wl_brother)
        plot_wl((-9, 9), 30, wl_sister, wl_brother; aspect_ratio=:equal, legend = false, size = (600, 600))
    end every 5

    wl_sister = copy(wl_sister_origin)
    wl_brother = copy(wl_brother_origin)
    anim_brother = @animate for idx ∈ 2:length(wl_brother)
        x⁰, x¹ = wl_brother[idx]
        L_wl!(x¹ / x⁰, wl_sister, wl_brother)
        shift!(wl_brother[idx], wl_sister, wl_brother)
        plot_wl((-9, 9), 30, wl_sister, wl_brother; aspect_ratio=:equal, legend = false, size = (600, 600))
    end every 5

    cd(@__DIR__)
    gif(anim_sister, "sister_perspective.gif")
    gif(anim_brother, "brother_perspective.gif")
    rm(anim_sister.dir; force = true, recursive = true)
    rm(anim_brother.dir; force = true, recursive = true)
end
main()
