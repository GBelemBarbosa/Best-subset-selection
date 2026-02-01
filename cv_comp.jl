# Terminal-running version of new_test.pluto.jl
# Converted from Pluto notebook to standard Julia script

@info "Loading packages..."
flush(stdout)
using Random
using Distributions
using LinearAlgebra
using BenchmarkTools, Profile, TimerOutputs
using Plots, StatsPlots, Plots.PlotMeasures
using LaTeXStrings
using ThreadsX

# Try to load RCall for L0Learn comparison
L0LEARN_AVAILABLE = false
try
    @eval using RCall
    rcall(:eval, rparse(".libPaths(c('~/R_libs', .libPaths()))"))
    rcall(:library, "L0Learn")
    global L0LEARN_AVAILABLE = true
    @info "L0Learn R package loaded successfully"
catch e
    @warn "L0Learn R package not available: $e"
end

include("l0learn_julia.jl")
@info "Packages loaded successfully"
flush(stdout)

# ============================================================================
# Parameters (can be overridden via command-line arguments)
# Usage: julia script.jl [corr] [ρ] [p] [SNR] [k⃰] [ns_start:ns_step:ns_end] [algo] [T]
# Example: julia script.jl const 0.9 1000 5 20 100:100:1000 CDSS 10
# algo options: CDSS, SPG, SPGpCDSS
# ============================================================================

# Defaults
corr = length(ARGS) >= 1 ? ARGS[1] : "const"
ρ = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.9
p = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1000
SNR = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 5.0
k⃰ = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 20

# Parse ns range (format: start:step:end)
if length(ARGS) >= 6
    ns_parts = split(ARGS[6], ":")
    ns_start = parse(Int, ns_parts[1])
    ns_step = parse(Int, ns_parts[2])
    ns_end = parse(Int, ns_parts[3])
    ns_range = ns_start:ns_step:ns_end
else
    ns_range = 100:100:1000  # Default
end

# algo options: CDSS, SPG, SPGpCDSS, L0Learn
algo_name = length(ARGS) >= 7 ? ARGS[7] : "CDSS"
use_SPG = algo_name in ["SPG", "SPGpCDSS"]  # SPG step logic
T = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 10

@info "Parameters" corr ρ p SNR k⃰ ns_range algo = algo_name use_SPG T
flush(stdout)

kₘₐₓ = 1000
ϵ = 10^-7

# ============================================================================
# Variable Generation
# ============================================================================

function variables(; corr="exp", ρ=0.9, n=250, p=1000, SNR=5, k⃰=20)
    Σ = corr == "exp" ? [ρ^abs(i - j) for i = 1:p, j = 1:p] : [1 - (1 - ρ) * (i != j) for i = 1:p, j = 1:p]
    d = MvNormal(zeros(p), Σ)
    X = rand(d, n)'
    for i = 1:p
        X[:, i] /= norm(view(X, :, i))
    end
    β⃰ = [1.0 * (mod(i, Int(p / k⃰)) == 0) for i = 1:p]

    σ = sqrt(norm(X * β⃰)^2 / (n * SNR))

    y = X * β⃰ + randn(n) * σ
    yval = X * β⃰ + randn(n) * σ

    XTX = X'X

    return X, y, yval, XTX, p, β⃰, k⃰
end

# ============================================================================
# Function Definitions
# ============================================================================

function funcs(X, y, yval, XTX, λ₀)
    HT = sqrt(2 * λ₀)
    r!(r, β) = (mul!(r, X, β); r .-= y)
    r(β) = X * β - y
    f(r) = norm(r)^2 / 2
    h(β) = λ₀ * norm(β, 0)
    F(r, β) = f(r) + h(β)
    ∇f!(∇f, r) = mul!(∇f, X', r)
    ∇f(r) = X'r
    proxl0(x) = (abs(x) >= HT) * x
    proxl0(x, τ) = (abs.(x) .>= sqrt(2λ₀ * τ)) .* x
    proxl0VM(x, Uₖ) = (abs.(x) .>= sqrt.(2λ₀ .* Uₖ)) .* x

    # In-place versions for reduced allocations
    function proxl0!(out, x, τ)
        thresh = sqrt(2λ₀ * τ)
        @inbounds @simd for i in eachindex(out)
            out[i] = abs(x[i]) >= thresh ? x[i] : zero(eltype(x))
        end
        return out
    end

    function proxl0VM!(out, x, Uₖ)
        @inbounds @simd for i in eachindex(out)
            out[i] = abs(x[i]) >= sqrt(2λ₀ * Uₖ[i]) ? x[i] : zero(eltype(x))
        end
        return out
    end

    return r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, y, XTX
end

# ============================================================================
# VMSPG Algorithm
# ============================================================================

function VMSPG(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64), µ=10^-3)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, y, XTX = funcs

    n_vars = length(x⁰)

    # Pre-allocate all work arrays
    xᵏ = copy(x⁰)
    xᵏ⁻¹ = copy(x⁰)
    sᵏ = similar(x⁰)
    yᵏ = similar(x⁰)
    ∇fxᵏ = similar(x⁰)
    ∇fxᵏ⁻¹ = similar(x⁰)
    temp = similar(x⁰)
    Uₖ = similar(x⁰)
    Uₖ₋₁ = similar(x⁰)

    rᵏ = r(xᵏ)
    ∇f!(∇fxᵏ, rᵏ)
    Fxᵏ⁻¹ = Fxᵏ = F(rᵏ, xᵏ)

    # Initial SPG step computation using BB formula
    # sᵏ = 10⁻⁵ * ∇f(x⁰), the artificial step in gradient direction
    # yᵏ = ∇f(x⁰) - ∇f(x⁰ - sᵏ) = X'X * sᵏ (gradient difference)
    ε = 10^-5
    XTX∇fxᵏ = XTX * ∇fxᵏ                  # X'X∇f(x⁰), used for yᵏ = ε * X'X∇f
    sᵏT∇fxᵏ = dot(∇fxᵏ, ∇fxᵏ)             # ⟨sᵏ, ∇f⟩ / ε = ‖∇f‖²
    yᵏTsᵏ = ε * dot(XTX∇fxᵏ, ∇fxᵏ)        # ⟨yᵏ, sᵏ⟩ = ε² * ⟨X'X∇f, ∇f⟩
    nyᵏ = ε^2 * dot(XTX∇fxᵏ, XTX∇fxᵏ)     # ‖yᵏ‖² = ε² * ‖X'X∇f‖²

    # BB1: γₖ¹ = ⟨sᵏ,sᵏ⟩/⟨yᵏ,sᵏ⟩ = ε²‖∇f‖² / (ε²⟨X'X∇f,∇f⟩) = ‖∇f‖²/⟨X'X∇f,∇f⟩
    γₖ¹ = sᵏT∇fxᵏ * ε / yᵏTsᵏ
    # BB2: γₖ² = ⟨yᵏ,sᵏ⟩/⟨yᵏ,yᵏ⟩ = ε²⟨X'X∇f,∇f⟩ / (ε²‖X'X∇f‖²)
    γₖ² = yᵏTsᵏ / nyᵏ
    nsᵏ = γₖ¹

    # Initialize Uₖ (variable metric) in-place
    # Uₖᵢ = sᵏᵢ² / (sᵏᵢ * yᵏᵢ) = ε∇fᵢ / (X'X∇f)ᵢ, clamped to [γₖ², γₖ¹]
    @inbounds for i in eachindex(Uₖ)
        val = ε * ∇fxᵏ[i] / XTX∇fxᵏ[i]
        Uₖ[i] = min(max(val, γₖ²), γₖ¹)
    end
    copyto!(Uₖ₋₁, Uₖ)

    lastₘ = fill(Fxᵏ, m)

    for k = 1:kₘₐₓ
        Fxₗ₍ₖ₎ = maximum(lastₘ)

        while true
            # xᵏ = proxl0VM(xᵏ⁻¹ - Uₖ .* ∇fxᵏ, Uₖ)
            temp .= xᵏ⁻¹ .- Uₖ .* ∇fxᵏ
            proxl0VM!(xᵏ, temp, Uₖ)
            r!(rᵏ, xᵏ)
            Fxᵏ = F(rᵏ, xᵏ)
            sᵏ .= xᵏ .- xᵏ⁻¹

            # Compute dot(sᵏ, sᵏ ./ Uₖ) without allocation
            ss_div_U = zero(eltype(sᵏ))
            @inbounds @simd for i in eachindex(sᵏ)
                ss_div_U += sᵏ[i] * sᵏ[i] / Uₖ[i]
            end

            if Fxᵏ + δ * ss_div_U / 2 <= Fxₗ₍ₖ₎
                break
            end

            BLAS.scal!(τ, Uₖ)

            # Check bounds without allocation
            should_break = false
            @inbounds for i in eachindex(Uₖ)
                if isnan(Uₖ[i]) || Uₖ[i] < γₘᵢₙ
                    should_break = true
                    break
                end
            end
            if should_break
                break
            end
        end

        if abs(Fxᵏ⁻¹ - Fxᵏ) / Fxᵏ <= ϵ
            return xᵏ, k
        end

        popfirst!(lastₘ)
        push!(lastₘ, Fxᵏ)
        copyto!(xᵏ⁻¹, xᵏ)
        Fxᵏ⁻¹ = Fxᵏ
        copyto!(∇fxᵏ⁻¹, ∇fxᵏ)
        ∇f!(∇fxᵏ, rᵏ)
        nsᵏ = dot(sᵏ, sᵏ)
        yᵏ .= ∇fxᵏ .- ∇fxᵏ⁻¹
        nyᵏ = dot(yᵏ, yᵏ)
        yᵏTsᵏ = dot(yᵏ, sᵏ)
        γₖ¹ = yᵏTsᵏ > 0 ? nsᵏ / yᵏTsᵏ : sqrt(nsᵏ / nyᵏ)
        γₖ² = yᵏTsᵏ > 0 ? yᵏTsᵏ / nyᵏ : 1 / γₖ¹

        # Update Uₖ in-place with buffer swap
        Uₖ₋₁, Uₖ = Uₖ, Uₖ₋₁
        @inbounds @simd for i in eachindex(Uₖ)
            val = (sᵏ[i] * sᵏ[i] + µ * Uₖ₋₁[i]) / (sᵏ[i] * yᵏ[i] + µ)
            Uₖ[i] = min(max(val, γₖ²), γₖ¹)
        end
    end

    return xᵏ, kₘₐₓ
end

# ============================================================================
# SPGH Algorithm
# ============================================================================

function SPGH(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64))
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, y, XTX = funcs

    # Pre-allocate all work arrays
    xᵏ = copy(x⁰)
    xᵏ⁻¹ = copy(x⁰)
    sᵏ = similar(x⁰)
    yᵏ = similar(x⁰)
    ∇fxᵏ = similar(x⁰)
    ∇fxᵏ⁻¹ = similar(x⁰)
    temp = similar(x⁰)

    rᵏ = r(xᵏ)
    ∇f!(∇fxᵏ, rᵏ)
    Fxᵏ⁻¹ = Fxᵏ = F(rᵏ, xᵏ)

    # Initial SPG step computation using BB formula
    # sᵏ = 10⁻⁵ * ∇f(x⁰), yᵏ = ε * X'X∇f(x⁰)
    ε = 10^-5
    XTX∇fxᵏ = XTX * ∇fxᵏ                  # X'X∇f(x⁰)
    sᵏT∇fxᵏ = dot(∇fxᵏ, ∇fxᵏ)             # ‖∇f‖²
    yᵏTsᵏ = ε * dot(XTX∇fxᵏ, ∇fxᵏ)        # ⟨yᵏ, sᵏ⟩
    nyᵏ = ε^2 * dot(XTX∇fxᵏ, XTX∇fxᵏ)     # ‖yᵏ‖²

    # BB1: γₖ¹ = ⟨sᵏ,sᵏ⟩/⟨yᵏ,sᵏ⟩, BB2: γₖ² = ⟨yᵏ,sᵏ⟩/⟨yᵏ,yᵏ⟩
    γₖ¹ = sᵏT∇fxᵏ * ε / yᵏTsᵏ
    γₖ² = yᵏTsᵏ / nyᵏ
    nsᵏ = γₖ¹
    # Hybrid selection: use BB2 if γₖ¹ < 2γₖ², else use midpoint
    γₖ = γₖ¹ < 2 * γₖ² ? γₖ² : γₖ¹ - γₖ² / 2
    lastₘ = fill(Fxᵏ, m)

    for k = 1:kₘₐₓ
        Fxₗ₍ₖ₎ = maximum(lastₘ)

        while true
            # xᵏ = proxl0(xᵏ⁻¹ - γₖ * ∇fxᵏ, γₖ)
            temp .= xᵏ⁻¹ .- γₖ .* ∇fxᵏ
            proxl0!(xᵏ, temp, γₖ)
            r!(rᵏ, xᵏ)
            Fxᵏ = F(rᵏ, xᵏ)
            sᵏ .= xᵏ .- xᵏ⁻¹
            nsᵏ = dot(sᵏ, sᵏ)

            if Fxᵏ + δ * nsᵏ / (2 * γₖ) <= Fxₗ₍ₖ₎
                break
            end

            γₖ *= τ

            if isnan(γₖ) || γₖ < γₘᵢₙ
                break
            end
        end

        if abs(Fxᵏ⁻¹ - Fxᵏ) / Fxᵏ <= ϵ
            return xᵏ, k
        end

        popfirst!(lastₘ)
        push!(lastₘ, Fxᵏ)
        copyto!(xᵏ⁻¹, xᵏ)
        Fxᵏ⁻¹ = Fxᵏ
        copyto!(∇fxᵏ⁻¹, ∇fxᵏ)
        ∇f!(∇fxᵏ, rᵏ)
        yᵏ .= ∇fxᵏ .- ∇fxᵏ⁻¹
        nyᵏ = dot(yᵏ, yᵏ)
        yᵏTsᵏ = dot(yᵏ, sᵏ)
        γₖ¹ = yᵏTsᵏ > 0 ? nsᵏ / yᵏTsᵏ : sqrt(nsᵏ / nyᵏ)
        γₖ² = yᵏTsᵏ > 0 ? yᵏTsᵏ / nyᵏ : 1 / γₖ¹
        γₖ = γₖ¹ < 2 * γₖ² ? γₖ² : γₖ¹ - γₖ² / 2
    end

    return xᵏ, kₘₐₓ
end

# ============================================================================
# SPG Algorithm
# ============================================================================

function SPG(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64), γₖ=0.0, kwargs...)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, y, XTX = funcs

    # Pre-allocate all work arrays
    xᵏ = copy(x⁰)
    xᵏ⁻¹ = copy(x⁰)
    sᵏ = similar(x⁰)
    yᵏ = similar(x⁰)
    ∇fxᵏ = similar(x⁰)
    ∇fxᵏ⁻¹ = similar(x⁰)
    temp = similar(x⁰)

    rᵏ = r(xᵏ)
    ∇f!(∇fxᵏ, rᵏ)
    Fxᵏ⁻¹ = Fxᵏ = F(rᵏ, xᵏ)

    if iszero(γₖ)
        # Initial SPG step via BB1: γₖ = ⟨sᵏ,sᵏ⟩/⟨yᵏ,sᵏ⟩
        # sᵏ = ε∇f, yᵏ = ∇fxᵏ + X'(y - X(β - ε∇f)) = εX'X∇f
        # γₖ = ε‖∇f‖² / ⟨yᵏ, ∇f⟩ = ε‖∇f‖² / (ε⟨X'X∇f,∇f⟩)
        # ε = 10^-5
        XTX∇fxᵏ = XTX * ∇fxᵏ
        γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
    end
    #@info "Initial γₖ (SPG)" γₖ
    nsᵏ = γₖ
    lastₘ = fill(Fxᵏ, m)

    for k = 1:kₘₐₓ
        Fxₗ₍ₖ₎ = maximum(lastₘ)

        while true
            # xᵏ = proxl0(xᵏ⁻¹ - γₖ * ∇fxᵏ, γₖ)
            temp .= xᵏ⁻¹ .- γₖ .* ∇fxᵏ
            proxl0!(xᵏ, temp, γₖ)
            r!(rᵏ, xᵏ)
            Fxᵏ = F(rᵏ, xᵏ)
            sᵏ .= xᵏ .- xᵏ⁻¹
            nsᵏ = dot(sᵏ, sᵏ)

            if Fxᵏ + δ * nsᵏ / (2 * γₖ) <= Fxₗ₍ₖ₎
                break
            end

            γₖ *= τ

            if isnan(γₖ) || γₖ < γₘᵢₙ
                break
            end
        end

        if abs(Fxᵏ⁻¹ - Fxᵏ) / Fxᵏ <= ϵ
            return xᵏ, k
        end

        popfirst!(lastₘ)
        push!(lastₘ, Fxᵏ)
        copyto!(xᵏ⁻¹, xᵏ)
        Fxᵏ⁻¹ = Fxᵏ
        copyto!(∇fxᵏ⁻¹, ∇fxᵏ)
        ∇f!(∇fxᵏ, rᵏ)

        yᵏ .= ∇fxᵏ .- ∇fxᵏ⁻¹
        γₖ = nsᵏ / dot(yᵏ, sᵏ)
        if γₖ > γₘₐₓ || γₖ < γₘᵢₙ
            γₖ = sqrt(nsᵏ / dot(yᵏ, yᵏ))
        end
    end

    return xᵏ, kₘₐₓ
end

# ============================================================================
# CDSS Algorithm with Active Set Convergence
# Based on L0Learn paper: Hazimeh & Mazumder (2018)
# Active Set Strategy: Full cycles until support stabilizes, then active set only,
# then verify with CW minimum check on coordinates outside support.
# ============================================================================

function CDSS(x⁰, funcs; sortperc=1 / 4, ActiveSetNum=10, kwargs...)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, y, XTX = funcs

    xᵏ = copy(x⁰)
    rᵏ = -r(xᵏ)
    Fxᵏ⁻¹ = F(rᵏ, xᵏ)

    n_vars = length(x⁰)
    ksort = round(Int64, n_vars * sortperc)

    # Initial greedy ordering based on correlations (computed ONCE)
    greedy = partialsortperm(abs.(∇f(rᵏ)), 1:ksort, rev=true)
    greedy = vcat(greedy, setdiff(1:n_vars, greedy))

    # Active set tracking (per L0Learn's RestrictSupport)
    xᵏ⁻¹ = copy(xᵏ)
    SameSuppCounter = 0
    Stabilized = false
    Order = greedy  # Current sweep order

    for k = 1:kₘₐₓ
        # RestrictSupport: check if support is same as previous iteration (per L0Learn)
        if !Stabilized
            if has_same_support(xᵏ, xᵏ⁻¹)
                SameSuppCounter += 1
                if SameSuppCounter == ActiveSetNum - 1
                    # Switch to active set mode (nonzero indices only)
                    Order = findall(!iszero, xᵏ)
                    Stabilized = true
                end
            else
                SameSuppCounter = 0
            end
        end

        copyto!(xᵏ⁻¹, xᵏ)

        @inbounds for i in Order
            xi = proxl0(dot(rᵏ, view(X, :, i)) + xᵏ[i])
            if xi != xᵏ[i]
                BLAS.axpy!(xᵏ[i] - xi, view(X, :, i), rᵏ)
                xᵏ[i] = xi
            end
        end

        Fxᵏ = F(rᵏ, xᵏ)
        if (Fxᵏ⁻¹ - Fxᵏ) / Fxᵏ <= ϵ
            return xᵏ, k
        end

        Fxᵏ⁻¹ = Fxᵏ
    end

    return xᵏ, kₘₐₓ
end

# Helper function matching L0Learn's has_same_support
function has_same_support(x, y)
    for i in eachindex(x)
        if iszero(x[i]) != iszero(y[i])
            return false
        end
    end
    return true
end

# ============================================================================
# SPG + CDSS Combined
# ============================================================================

function SPGpCDSS(x⁰, funcs; γₖ=0.0, kwargs...)
    x, k = SPG(x⁰, funcs; γₖ=γₖ, kwargs...)
    x, k2 = CDSS(x, funcs; kwargs...)

    return x, k + k2
end

# ============================================================================
# ============================================================================
# L0Learn Helper (R integration)
# ============================================================================

function solve_l0_single_lambda_R(lambda_val, x_init, X_train, y_train; maxSuppSize=size(X_train, 2), algorithm="CD")
    try
        RCall.globalEnv[:X_train] = collect(X_train)
        RCall.globalEnv[:y_train] = collect(y_train)
        RCall.globalEnv[:lambda_val] = lambda_val
        RCall.globalEnv[:maxSuppSize] = maxSuppSize
        RCall.globalEnv[:algo] = algorithm
        
        script = raw"""
            .libPaths(unique(c("/home/gbelem/R/library", .libPaths())))
            suppressMessages(library(L0Learn))
            n <- nrow(X_train)
            lambda_scaled <- lambda_val / n
            fit <- L0Learn.fit(X_train, y_train, penalty="L0", algorithm=algo, lambdaGrid=list(c(lambda_scaled)), 
                               maxSuppSize=maxSuppSize, intercept=FALSE)
            if (length(fit$lambda[[1]]) > 0) {
                as.numeric(coef(fit, lambda=lambda_scaled, gamma=0))
            } else {
                numeric(ncol(X_train))
            }
        """
        return RCall.rcopy(RCall.reval(script))
    catch e
        @warn "L0Learn R solve failed: $e. Falling back to Julia implementation."
        return l0learn_fit_julia(X_train, y_train, lambda_val; beta_init=x_init)
    end
end

function L0LearnStep(xᵏ, funcs; lambda_val=nothing, X_data=nothing, y_data=nothing, kwargs...)
    if lambda_val === nothing
        return xᵏ, 0
    end
    
    # In cv_comp.jl, funcs has X at 11, y at 12, XTX at 13
    X = X_data !== nothing ? X_data : (length(funcs) >= 11 ? funcs[11] : nothing)
    y = y_data !== nothing ? y_data : (length(funcs) >= 12 ? funcs[12] : nothing)
    
    if X === nothing || y === nothing
        @warn "L0LearnStep: Missing X or y"
        return xᵏ, 0
    end

    if L0LEARN_AVAILABLE
        beta_new = solve_l0_single_lambda_R(lambda_val, xᵏ, X, y; maxSuppSize=size(X, 2))
    else
        beta_new = l0learn_fit_julia(X, y, lambda_val; beta_init=xᵏ)
    end
    
    return beta_new, 1
end

# ============================================================================
# PSI1 Algorithm
# ============================================================================

function PSI1(xˡ, funcs)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, y, XTX = funcs

    r⃰ = -X'r(xˡ)

    # Cache index arrays to avoid repeated allocation
    nonzero_indices = findall(!iszero, xˡ)
    zero_indices = findall(iszero, xˡ)

    for i in nonzero_indices
        jₘₐₓ = 0
        v⃰ₘₐₓ = 0.0

        @inbounds for j in zero_indices
            v⃰ = proxl0(r⃰[j] + XTX[i, j] * xˡ[i])

            if abs(v⃰) > abs(v⃰ₘₐₓ)
                v⃰ₘₐₓ = v⃰
                jₘₐₓ = j
            end
        end

        if abs(v⃰ₘₐₓ) > abs(xˡ[i])
            xˡ[i] = 0.0
            xˡ[jₘₐₓ] = v⃰ₘₐₓ
            return xˡ, false
        end
    end

    return xˡ, true
end

# ============================================================================
function SolverPSI1(solver, x⁰, funcs; γₖ=0.0, max_psi1_fails=50, kwargs...)
    β = x⁰
    kᵢ = kₒ = 0
    isPSI1 = false

    # Cycle detection: track support patterns
    seen_supports = Set{UInt64}()
    consecutive_fails = 0

    while !isPSI1 && kₒ < kₘₐₓ
        kₒ += 1
        β, k = solver(β, funcs; γₖ=γₖ, kwargs...)
        kᵢ += k
        γₖ = 0.0 # Only the first call uses the pre-computed validation γₖ
        β, isPSI1 = PSI1(β, funcs)

        if !isPSI1
            consecutive_fails += 1
            # Hash the current support pattern for cycle detection
            support_hash = hash(findall(!iszero, β))
            if support_hash in seen_supports
                break
            end
            push!(seen_supports, support_hash)

            # Also break if too many consecutive PSI1 failures
            if consecutive_fails >= max_psi1_fails
                break
            end
        else
            consecutive_fails = 0
            empty!(seen_supports)
        end
    end

    return β, kᵢ, kₒ
end

# ============================================================================
# Cross Validation
# ============================================================================

function cross_validation(solver, vars; λ_min_ratio=10^-5, SPG=false, stagnation_handling=true)
    X, y, yval, XTX, p, β⃰, k⃰ = vars
    suppsim(β) = count(i -> !iszero(β⃰[i]) && !iszero(β[i]), 1:p) / max(k⃰, norm(β, 0))
    predval(β) = norm(X * β .- yval)^2 / norm(yval)^2

    β = β_best = zeros(p)
    best = best_λ = Inf

    ∇fxᵏ = -X'y

    if SPG
        # ε = 10^-5, sᵏ = ε∇f
        # yᵏ = ∇fxᵏ + X'(y - X(β - ε∇f)) = εX'X∇f
        # γₖ = ε‖∇f‖² / ⟨yᵏ, ∇f⟩ = ε‖∇f‖² / (ε⟨X'X∇f,∇f⟩)
        XTX∇fxᵏ = X' * X * ∇fxᵏ
        γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
    else
        γₖ = 1.0
    end

    # λ formula always uses γₖ (computed SPG val or 1.0)
    λ = 1.01 * γₖ * ThreadsX.maximum(abs(∇fxᵏ[j]) for j = 1:p)^2 / 2
    λ_min = λ * λ_min_ratio

    i = 1
    stagnant_count = 0
    prev_computed_λ = λ  # Track what formula computes (before any bump)

    while λ > λ_min && norm(β, 0) != p
        # Run solver - only pass γₖ if we are in SPG mode
        β, kᵢ, kₒ = solver(β, funcs(X, y, yval, XTX, λ); γₖ=SPG ? γₖ : 0.0, lambda_val=λ, X_data=X, y_data=y)

        mse = norm(yval .- X * β)
        if mse < best
            β_best = copy(β)
            best = mse
            best_λ = λ
        end

        if norm(β, 0) == p
            break
        end

        if SPG
            # ε = 10^-5, sᵏ = ε∇f
            # yᵏ = ∇fxᵏ + X'(y - X(β - ε∇f)) = εX'X∇f
            ∇fxᵏ = X' * (X * β - y)
            XTX∇fxᵏ = X' * X * ∇fxᵏ
            γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
            #@info "Update γₖ (Cross Validation)" γₖ
        end

        # Compute new λ from formula - with min to previous λ for consistency
        # The 0.9 multiplier is OUTSIDE the min for monotonic decrease
        raw_λ = norm(β, 0) != p ? γₖ * ThreadsX.maximum(abs(dot(view(X, :, j), X * β - y)) for j = 1:p if iszero(β[j]))^2 / 2 : 0.0
        computed_λ = 0.9 * min(prev_computed_λ, raw_λ)

        if stagnation_handling
            # Check if formula is stagnant (computing same value repeatedly)
            if abs(computed_λ - prev_computed_λ) / max(prev_computed_λ, eps()) < 0.01
                stagnant_count += 1
                # When stagnant, never increase λ (opposite of inverse CV)
                new_λ = computed_λ > λ ? λ : computed_λ
                if stagnant_count >= 3
                    new_λ = 0.9 * λ  # Force monotonic decrease
                    stagnant_count = 0
                end
            else
                stagnant_count = 0
                new_λ = computed_λ
            end
            prev_computed_λ = computed_λ
            λ = new_λ
        else
            λ = computed_λ  # Simple monotonic decrease
        end
        i += 1
    end

    # Final refinement with best λ
    β_best_path, kᵢ, kₒ = solver(β_best, funcs(X, y, yval, XTX, best_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=best_λ, X_data=X, y_data=y)
    β_best_zero, kᵢ, kₒ = solver(zeros(p), funcs(X, y, yval, XTX, best_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=best_λ, X_data=X, y_data=y)
    
    zero_won = predval(β_best_zero) < predval(β_best_path)
    zero_improves_sim = zero_won && (suppsim(β_best_zero) > suppsim(β_best_path))
    
    β_ref = zero_won ? β_best_zero : β_best_path

    return β_ref, best_λ, norm(β_ref, 0), predval(β_ref), suppsim(β_ref), norm(β_ref - β⃰, Inf), zero_won, zero_improves_sim
end

# ============================================================================
# Inverse Cross Validation (lambda grows instead of shrinks)
# ============================================================================

function inverse_cross_validation(solver, vars; λ_max_ratio=1e30, SPG=false, stagnation_handling=true)
    X, y, yval, XTX, p, β⃰, k⃰ = vars
    suppsim(β) = count(i -> !iszero(β⃰[i]) && !iszero(β[i]), 1:p) / max(k⃰, norm(β, 0))
    predval(β) = norm(X * β .- yval)^2 / norm(yval)^2

    β = β_best = zeros(p)
    best = best_λ = Inf

    # Compute initial residual correlations (residual = y when β = 0)
    ∇fxᵏ = -X'y

    # Initial λ: use max_corr for inverse CV (we start low and go high)
    # Using min_corr fails for exp correlation where min_corr ≈ 0
    min_corr = ThreadsX.minimum(abs(∇fxᵏ[j]) for j = 1:p)

    if SPG
        # ε = 10^-5, sᵏ = ε∇f
        # yᵏ = ∇fxᵏ + X'(y - X(β - ε∇f)) = εX'X∇f
        # γₖ = ε‖∇f‖² / ⟨yᵏ, ∇f⟩ = ε‖∇f‖² / (ε⟨X'X∇f,∇f⟩)
        XTX∇fxᵏ = X' * X * ∇fxᵏ
        γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
    else
        γₖ = 1.0
    end

    # Initial λ for inverse CV: start SMALL to get high support, then increase
    # Use a fraction of max_corr^2 to start in the dense regime
    λ = (1.01^-1) * γₖ * min_corr^2 / 2  # Small multiplier to ensure high initial support

    # --- PRE-LOOP WARMUP: Reduce λ until we get non-zero support ---
    # For L0Learn with const correlation, the initial λ may be too large.
    β_warmup, _, _ = solver(zeros(p), funcs(X, y, yval, XTX, λ); γₖ=SPG ? γₖ : 0.0, lambda_val=λ, X_data=X, y_data=y)
    warmup_iters = 0
    while iszero(norm(β_warmup, 0)) && warmup_iters < 50
        λ *= 0.9
        β_warmup, _, _ = solver(zeros(p), funcs(X, y, yval, XTX, λ); γₖ=SPG ? γₖ : 0.0, lambda_val=λ, X_data=X, y_data=y)
        warmup_iters += 1
    end
    if warmup_iters >= 50 && iszero(norm(β_warmup, 0))
        @warn "Inverse CV warmup failed: could not find non-zero beta after 50 iterations (final λ=$λ)"
    end
    β = β_warmup  # Start main loop with the warmed-up solution
    # --- END WARMUP ---

    λ_max = λ * λ_max_ratio

    i = 1
    stagnant_count = 0
    prev_λ = λ

    while λ < λ_max && (!iszero(norm(β, 0)) || i == 1)
        # Ensure λ stays positive
        λ = max(λ, eps())

        # Run solver
        β, kᵢ, kₒ = solver(β, funcs(X, y, yval, XTX, λ); γₖ=SPG ? γₖ : 0.0, lambda_val=λ, X_data=X, y_data=y)

        mse = norm(yval .- X * β)
        if mse < best
            β_best = copy(β)
            best = mse
            best_λ = λ
        end

        # Stop if support is zero
        if norm(β, 0) == 0
            break
        end

        if SPG
            ∇fxᵏ = X' * (X * β - y)
            XTX∇fxᵏ = X' * X * ∇fxᵏ
            γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
        else
            γₖ = 1.0
        end

        # Compute new λ from formula - with max to previous λ for monotonic increase
        # The 0.9^-1 multiplier is OUTSIDE the max
        raw_λ = ThreadsX.minimum(abs(β[j] - γₖ * dot(view(X, :, j), X * β - y)) for j = 1:p if !iszero(β[j]))^2 / (2 * γₖ)
        computed_λ = (0.9^-1) * max(prev_λ, raw_λ)

        if stagnation_handling
            # Stagnation detection: if λ isn't increasing, force it up
            if abs(computed_λ - prev_λ) / max(prev_λ, eps()) < 0.01
                stagnant_count += 1
                new_λ = computed_λ < λ ? λ : computed_λ
                if stagnant_count >= 3
                    new_λ = 1.1 * λ  # Force monotonic increase
                    stagnant_count = 0
                end
            else
                stagnant_count = 0
                new_λ = computed_λ
            end
            prev_λ = λ
            λ = new_λ
        else
            λ = computed_λ  # Simple monotonic increase
        end
        i += 1
    end


    # Final refinement with best λ
    β_best_path, kᵢ, kₒ = solver(β_best, funcs(X, y, yval, XTX, best_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=best_λ, X_data=X, y_data=y)
    β_best_zero, kᵢ, kₒ = solver(zeros(p), funcs(X, y, yval, XTX, best_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=best_λ, X_data=X, y_data=y)
    
    zero_won = predval(β_best_zero) < predval(β_best_path)
    zero_improves_sim = zero_won && (suppsim(β_best_zero) > suppsim(β_best_path))
    
    β_ref = zero_won ? β_best_zero : β_best_path

    return β_ref, best_λ, norm(β_ref, 0), predval(β_ref), suppsim(β_ref), norm(β_ref - β⃰, Inf), zero_won, zero_improves_sim
end

# ============================================================================
# Main Experiment
# ============================================================================

# ============================================================================
# Smart Adaptive Cross Validation (Valley Descent)
# ============================================================================

function smart_adaptive_cross_validation(solver, vars;
    λ_min_ratio=10^-5,
    λ_max_ratio=floatmax(),
    SPG=false,
    stagnation_handling=true)

    X, y, yval, XTX, p, β⃰, k⃰ = vars
    suppsim(β) = count(i -> !iszero(β⃰[i]) && !iszero(β[i]), 1:p) / max(k⃰, norm(β, 0))
    predval(β) = norm(X * β .- yval)^2 / norm(yval)^2
    calc_mse(β) = norm(yval .- X * β)

    # --- 1. PROBE PHASE ---
    ∇fxᵏ = -X'y
    correlations = abs.(∇fxᵏ)
    max_corr = ThreadsX.maximum(correlations)
    min_corr = ThreadsX.minimum(correlations)

    if SPG
        XTX∇fxᵏ = X' * X * ∇fxᵏ
        γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
    else
        γₖ = 1.0
    end

    λ_low = (1.01^-1) * γₖ * min_corr^2 / 2
    λ_high = 1.01 * γₖ * max_corr^2 / 2

    # Probe Low
    β_low, _, _ = solver(zeros(p), funcs(X, y, yval, XTX, λ_low); γₖ=SPG ? γₖ : 0.0, lambda_val=λ_low, X_data=X, y_data=y)
    
    # --- PRE-LOOP WARMUP: Reduce λ_low until we get non-zero support ---
    # For L0Learn with const correlation, λ_low may still be too large.
    warmup_iters = 0
    while iszero(norm(β_low, 0)) && warmup_iters < 50
        λ_low *= 0.9
        β_low, _, _ = solver(zeros(p), funcs(X, y, yval, XTX, λ_low); γₖ=SPG ? γₖ : 0.0, lambda_val=λ_low, X_data=X, y_data=y)
        warmup_iters += 1
    end
    if warmup_iters >= 50 && iszero(norm(β_low, 0))
        @warn "Smart Adaptive CV warmup failed: could not find non-zero beta_low after 50 iterations (final λ_low=$λ_low)"
    end
    # --- END WARMUP ---
    
    mse_low = calc_mse(β_low)

    # Probe High
    β_high, _, _ = solver(zeros(p), funcs(X, y, yval, XTX, λ_high); γₖ=SPG ? γₖ : 0.0, lambda_val=λ_high, X_data=X, y_data=y)
    mse_high = calc_mse(β_high)

    # Select Best Start
    if mse_low < mse_high
        current_λ = λ_low
        current_β = β_low
        current_mse = mse_low
        direction = :increase
        λ_limit = λ_high * λ_max_ratio
    else
        current_λ = λ_high
        current_β = β_high
        current_mse = mse_high
        direction = :decrease
        λ_limit = λ_low * λ_min_ratio
    end

    best_mse = current_mse
    best_λ = current_λ
    best_β = copy(current_β)

    stagnant_count = 0
    prev_computed_λ = current_λ

    # --- 2. DESCENT PHASE ---
    i = 2
    # Oscillation detection
    direction_flip_count = 0

    while true
        # A. Propose Next Lambda
        if direction == :increase
            SUP = Int(norm(current_β, 0))
            rᵏ = X * current_β - y
            val = SUP == 0 ? 0.0 : ThreadsX.minimum(abs(current_β[j] - γₖ * dot(view(X, :, j), rᵏ)) for j = 1:p if !iszero(current_β[j]))
            raw_λ = val^2 / (2 * γₖ)
            # max with previous λ, THEN multiply by 0.9^-1
            computed_λ = (0.9^-1) * max(current_λ, raw_λ)
        else # :decrease
            SUP = Int(norm(current_β, 0))
            rᵏ = X * current_β - y
            val = SUP == p ? 0.0 : ThreadsX.maximum(abs(dot(view(X, :, j), rᵏ)) for j = 1:p if iszero(current_β[j]))
            raw_λ = γₖ * val^2 / 2
            # min with previous λ, THEN multiply by 0.9
            computed_λ = 0.9 * min(current_λ, raw_λ)
        end

        # Stagnation Handling
        if stagnation_handling
            if abs(computed_λ - prev_computed_λ) / max(prev_computed_λ, eps()) < 0.01
                stagnant_count += 1

                if direction == :increase
                    # Don't let it drop if we are increasing
                    computed_λ = computed_λ < current_λ ? current_λ : computed_λ
                    if stagnant_count >= 3
                        computed_λ = current_λ * 1.1 # Force up
                        stagnant_count = 0
                    end
                else # decrease
                    # Don't let it rise if we are decreasing
                    computed_λ = computed_λ > current_λ ? current_λ : computed_λ
                    if stagnant_count >= 3
                        computed_λ = current_λ * 0.9 # Force down
                        stagnant_count = 0
                    end
                end
            else
                stagnant_count = 0
            end
        end
        prev_computed_λ = computed_λ

        # Safety clamps
        if direction == :increase && computed_λ > λ_limit
            break
        end
        if direction == :decrease && computed_λ < λ_limit
            break
        end

        # B. Probe Next Step
        probe_λ = computed_λ
        probe_β, _, _ = solver(current_β, funcs(X, y, yval, XTX, probe_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=probe_λ, X_data=X, y_data=y)
        probe_mse = calc_mse(probe_β)

        # C. Check Improvement
        min_improvement = 0.01
        if probe_mse >= current_mse
            # Try switching direction
            alt_direction = (direction == :increase) ? :decrease : :increase

            # Propose Alt Lambda
            if alt_direction == :increase
                SUP = Int(norm(current_β, 0))
                rᵏ = X * current_β - y
                val = SUP == 0 ? 0.0 : ThreadsX.minimum(abs(current_β[j] - γₖ * dot(view(X, :, j), rᵏ)) for j = 1:p if !iszero(current_β[j]))
                raw_λ = val^2 / (2 * γₖ)
                alt_λ = (0.9^-1) * max(current_λ, raw_λ)
            else # :decrease
                SUP = Int(norm(current_β, 0))
                rᵏ = X * current_β - y
                val = SUP == p ? 0.0 : ThreadsX.maximum(abs(dot(view(X, :, j), rᵏ)) for j = 1:p if iszero(current_β[j]))
                raw_λ = γₖ * val^2 / 2
                alt_λ = 0.9 * min(current_λ, raw_λ)
            end

            alt_β, _, _ = solver(current_β, funcs(X, y, yval, XTX, alt_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=alt_λ, X_data=X, y_data=y)
            alt_mse = calc_mse(alt_β)

            # Pick whichever direction gives the smaller MSE
            if alt_mse < probe_mse * (1 - min_improvement)
                # Alt direction is better
                probe_λ = alt_λ
                probe_β = alt_β
                probe_mse = alt_mse
                direction = alt_direction

                # Oscillation check
                direction_flip_count += 1
                if direction_flip_count >= 2
                    # Oscillation detected (local minima), stopping
                    break
                end
                # Update λ_limit for new direction
                λ_limit = (direction == :increase) ? λ_high * λ_max_ratio : λ_low * λ_min_ratio

                # Reset stagnation on switch
                prev_computed_λ = probe_λ
            end
        end

        # Always accept and update
        current_λ = probe_λ
        current_β = probe_β
        current_mse = probe_mse
        if current_mse < best_mse
            best_mse = current_mse
            best_λ = current_λ
            best_β = copy(current_β)
        end

        if norm(current_β, 0) == 0
            break
        end

        # Update γₖ if SPG mode
        if SPG
            ∇fxᵏ = X' * (X * current_β - y)
            XTX∇fxᵏ = X' * X * ∇fxᵏ
            γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
        end
        i += 1
    end

    # Final refinement with best λ
    β_best_path, kᵢ, kₒ = solver(best_β, funcs(X, y, yval, XTX, best_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=best_λ, X_data=X, y_data=y)
    β_best_zero, kᵢ, kₒ = solver(zeros(p), funcs(X, y, yval, XTX, best_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=best_λ, X_data=X, y_data=y)
    
    zero_won = predval(β_best_zero) < predval(β_best_path)
    zero_improves_sim = zero_won && (suppsim(β_best_zero) > suppsim(β_best_path))
    
    β_ref = zero_won ? β_best_zero : β_best_path

    return β_ref, best_λ, norm(β_ref, 0), predval(β_ref), suppsim(β_ref), norm(β_ref - β⃰, Inf), zero_won, zero_improves_sim
end

function main()
    x⁰ = zeros(Float64, p)

    # Resolve algorithm function here (after definitions are loaded)
    algo_func = if algo_name == "SPG"
        SPG
    elseif algo_name == "SPGpCDSS"
        SPGpCDSS
    elseif algo_name == "L0Learn"
        L0LearnStep
    else
        CDSS
    end

    strategies = ["Regular CV", "Inverse CV", "Smart Adaptive"]
    @info "Experiment configuration" samples = ns_range trials = T strategies

    Predhist = zeros(length(ns_range), 3)
    SUPhist = zeros(length(ns_range), 3)
    Infhist = zeros(length(ns_range), 3)
    Simhist = zeros(length(ns_range), 3)
    Timehist = zeros(length(ns_range), 3)
    # Refinement stats: [strategy] -> ZeroWins, ZeroSimImprov
    ZWhist = zeros(Float64, length(ns_range), 3)
    ZIhist = zeros(Float64, length(ns_range), 3)

    # Early stopping tracking
    consecutive_perfect = 0
    last_completed_idx = 0
    early_stop = false

    for (t, n) in enumerate(ns_range)
        @info "Sample size n=$n" progress = "$(t)/$(length(ns_range))"
        flush(stdout)

        for i = 1:T
            vars = variables(corr=corr, ρ=ρ, n=n, p=p, SNR=SNR, k⃰=k⃰)
            X_curr, y_curr = vars[1], vars[2] # For L0LearnStep

            # 1. Regular CV (Standard cross_validation)
            trial_start = time()
            β, _, SUP, Pred, Sim, Infv, zw, zi = cross_validation((x, f; kwargs...) -> SolverPSI1(algo_func, x, f; y_data_top=y_curr, kwargs...), vars, SPG=use_SPG)
            trial_time = time() - trial_start
            SUPhist[t, 1] += SUP
            Predhist[t, 1] += Pred
            Simhist[t, 1] += Sim
            Infhist[t, 1] += Infv
            Timehist[t, 1] += trial_time
            ZWhist[t, 1] += zw ? 1.0 : 0.0
            ZIhist[t, 1] += zi ? 1.0 : 0.0
            @show "Regular" SUP Pred Sim Infv round(trial_time, digits=1)
            flush(stdout)

            # 2. Inverse CV (inverse_cross_validation)
            trial_start = time()
            β, _, SUP, Pred, Sim, Infv, zw, zi = inverse_cross_validation((x, f; kwargs...) -> SolverPSI1(algo_func, x, f; y_data_top=y_curr, kwargs...), vars, SPG=use_SPG)
            trial_time = time() - trial_start
            SUPhist[t, 2] += SUP
            Predhist[t, 2] += Pred
            Simhist[t, 2] += Sim
            Infhist[t, 2] += Infv
            Timehist[t, 2] += trial_time
            ZWhist[t, 2] += zw ? 1.0 : 0.0
            ZIhist[t, 2] += zi ? 1.0 : 0.0
            @show "Inverse" SUP Pred Sim Infv round(trial_time, digits=1)
            flush(stdout)

            # 3. Smart Adaptive CV
            trial_start = time()
            β, _, SUP, Pred, Sim, Infv, zw, zi = smart_adaptive_cross_validation((x, f; kwargs...) -> SolverPSI1(algo_func, x, f; y_data_top=y_curr, kwargs...), vars, SPG=use_SPG)
            trial_time = time() - trial_start
            SUPhist[t, 3] += SUP
            Predhist[t, 3] += Pred
            Simhist[t, 3] += Sim
            Infhist[t, 3] += Infv
            Timehist[t, 3] += trial_time
            ZWhist[t, 3] += zw ? 1.0 : 0.0
            ZIhist[t, 3] += zi ? 1.0 : 0.0
            @show "Adaptive" SUP Pred Sim Infv round(trial_time, digits=1)
            flush(stdout)
        end

        last_completed_idx = t

        println("\n--- n=$n Results (Avg over $T trials) ---")
        println("Strategy        | SUP   | Pred    | Sim   | Time (s) | Z-Wins | Z-Improv")
        println("----------------|-------|---------|-------|----------|--------|---------")
        println("Regular CV      | $(round(SUPhist[t, 1]/T, digits=1)) | $(round(Predhist[t, 1]/T, digits=4)) | $(round(Simhist[t, 1]/T, digits=3)) | $(round(Timehist[t, 1]/T, digits=1)) | $(Int(ZWhist[t, 1])) | $(Int(ZIhist[t, 1]))")
        println("Inverse CV      | $(round(SUPhist[t, 2]/T, digits=1)) | $(round(Predhist[t, 2]/T, digits=4)) | $(round(Simhist[t, 2]/T, digits=3)) | $(round(Timehist[t, 2]/T, digits=1)) | $(Int(ZWhist[t, 2])) | $(Int(ZIhist[t, 2]))")
        println("Smart Adaptive  | $(round(SUPhist[t, 3]/T, digits=1)) | $(round(Predhist[t, 3]/T, digits=4)) | $(round(Simhist[t, 3]/T, digits=3)) | $(round(Timehist[t, 3]/T, digits=1)) | $(Int(ZWhist[t, 3])) | $(Int(ZIhist[t, 3]))")
        println("----------------------------------------------------------------------\n")

        # Check early stopping: all strategies have perfect support recovery
        avg_sim = Simhist[t, :] ./ T
        all_perfect = all(avg_sim .>= 0.999)

        if all_perfect
            consecutive_perfect += 1
            @info "Perfect support recovery for all strategies" consecutive_count = consecutive_perfect
            if consecutive_perfect >= 2
                @info "Early stopping triggered: 2 consecutive n values with 100% support recovery"
                early_stop = true
                break
            end
        else
            consecutive_perfect = 0
        end
    end

    # Trim histograms to only completed n values
    ns_completed = collect(ns_range)[1:last_completed_idx]
    SUPhist = SUPhist[1:last_completed_idx, :]
    Predhist = Predhist[1:last_completed_idx, :]
    Simhist = Simhist[1:last_completed_idx, :]
    Infhist = Infhist[1:last_completed_idx, :]
    Timehist = Timehist[1:last_completed_idx, :]
    ZWhist = ZWhist[1:last_completed_idx, :]
    ZIhist = ZIhist[1:last_completed_idx, :]

    SUPhist ./= T
    Predhist ./= T
    Simhist ./= T
    Infhist ./= T
    Timehist ./= T
    ZWhist ./= T
    ZIhist ./= T

    # Plotting
    names = strategies |> permutedims
    display_name = algo_name == "L0Learn" ? "L0LearnPSI1" : algo_name
    plotname = "$(display_name)_Comparison_$(corr)-$(ρ)-$(p)-$(SNR)-$(k⃰)"

    # Global plot theme settings for "popy" look
    theme(:seaborn_bright)
    default(lw=3)

    println("\nGenerating comparison plots...")
    pPred = plot(ns_completed, Predhist, labels=names, xlabel=L"n", ylabel=L"\frac{\Vert Ax-b\Vert^2}{\Vert b\Vert^2}", left_margin=5mm, dpi=600)
    savefig(pPred, "Comp_Pred-$(plotname).png")

    pSim = plot(ns_completed, Simhist, labels=names, xlabel=L"n", ylabel=L"\frac{|Supp(x)\cap Supp(x^\dagger)|}{\max\{|Supp(x)|,k^\dagger\}}", left_margin=5mm, dpi=600)
    savefig(pSim, "Comp_Sim-$(plotname).png")

    pSUP = plot(ns_completed, SUPhist, labels=names, xlabel=L"n", ylabel=L"\Vert x\Vert_0", left_margin=5mm, dpi=600)
    savefig(pSUP, "Comp_SUP-$(plotname).png")

    pInf = plot(ns_completed, Infhist, labels=names, xlabel=L"n", ylabel=L"\Vert x-x^\dagger\Vert_\infty", left_margin=5mm, dpi=600)
    savefig(pInf, "Comp_Inf-$(plotname).png")

    pTime = plot(ns_completed, Timehist, labels=names, xlabel=L"n", ylabel="Execution Time (s)", left_margin=5mm, dpi=600, legend=:topleft)
    savefig(pTime, "Comp_Time-$(plotname).png")

    pRef = plot(ns_completed, ZWhist, labels=names, xlabel=L"n", ylabel="Rate (Wins / Improvs)", left_margin=5mm, top_margin=10mm, dpi=600, title="Zero-Start Wins (solid) vs \nSim Improvements (dashed)")
    plot!(pRef, ns_completed, ZIhist, labels="", linestyle=:dash, color=palette(:default)[1:3]')
    savefig(pRef, "Comp_Refinement-$(plotname).png")

    return SUPhist, Predhist, Simhist, Infhist, Timehist, ZWhist, ZIhist
end

SUPhist, Predhist, Simhist, Infhist = main()