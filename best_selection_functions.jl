# Terminal-running version of new_test.pluto.jl
# Converted from Pluto notebook to standard Julia script

@info "Loading packages..."
using Random
using Distributions
using LinearAlgebra
using BenchmarkTools, Profile, TimerOutputs
using Plots, StatsPlots, Plots.PlotMeasures
using LaTeXStrings
using ThreadsX
using Printf
include("l0learn_julia.jl")
# Try to load RCall for L0Learn comparison
L0LEARN_AVAILABLE = false
try
    @eval using RCall
    # Add local library path to check too
    rcall(:eval, rparse(".libPaths(c('~/R_libs', .libPaths()))"))
    rcall(:library, "L0Learn")
    global L0LEARN_AVAILABLE = true
    @info "L0Learn R package loaded successfully"
catch e
    @warn "L0Learn not available. Error: $e"
    @warn "Install with: R -e 'install.packages(\"L0Learn\")' and Julia Pkg.add(\"RCall\")"
end

@info "Packages loaded successfully"

# ============================================================================
# Parameters (can be overridden via command-line arguments)
# Usage: julia script.jl [corr] [ρ] [p] [SNR] [k⃰] [ns_start:ns_step:ns_end] [T]
# Example: julia script.jl const 0.9 1000 5 20 100:100:1000 10
# ============================================================================

# Defaults
corr = length(ARGS) >= 1 ? ARGS[1] : "exp"
ρ = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.9
n = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 250
p = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1000
k⃰ = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 20

# Parse SNR range (format: start:step:end)
if length(ARGS) >= 6
    snr_range_str = ARGS[6]
    snr_parts = split(ARGS[6], ":")
    snr_start = parse(Float64, snr_parts[1])
    snr_step = parse(Float64, snr_parts[2])
    snr_end = parse(Float64, snr_parts[3])
    snr_range = 10 .^ (snr_start:snr_step:snr_end)
else
    snr_range_str = "-2.0:0.5:2.0"
    snr_range = 10 .^ (-2:0.5:2)  # Default: 0.01 to 100
end

# Parse T (number of trials)
T = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 10

# Parse tridiagonal flag
is_tridiagonal = length(ARGS) >= 8 ? parse(Bool, ARGS[8]) : false

# Parse warmstart flag
use_warmstart = length(ARGS) >= 9 ? parse(Bool, ARGS[9]) : false

@info "Parameters" corr ρ n p k⃰ snr_range T is_tridiagonal use_warmstart L0LEARN_AVAILABLE
flush(stdout)

kₘₐₓ = 1000
ϵ = 10^-7

# Helper function to compute X' * X * x without allocating the full X' * X matrix
function XTX_mul(X, x)
    return X' * (X * x)
end

# ============================================================================
# Variable Generation
# ============================================================================

function variables(; corr="exp", ρ=0.9, n=250, p=1000, SNR=5, k⃰=20, is_tridiagonal=is_tridiagonal)
    Σ = corr == "exp" ? [ρ^abs(i - j) for i = 1:p, j = 1:p] : [1 - (1 - ρ) * (i != j) for i = 1:p, j = 1:p]
    d = MvNormal(zeros(p), Σ)
    X = rand(d, n)'
    
    # Enforce stepped full tridiagonal block structure if requested
    if is_tridiagonal
        ratio = p / n
        for i in 1:n
            start_col = floor(Int, (i - 2) * ratio) + 1
            end_col = floor(Int, (i + 1) * ratio)
            
            # Ensure we cover everything and don't go out of bounds
            start_col = max(1, start_col)
            end_col = min(p, end_col)
            
            for j in 1:p
                if j < start_col || j > end_col
                    X[i, j] = 0.0
                end
            end
        end
    end
    # End of tridiagonal modification
    for i = 1:p
        X[:, i] /= iszero(X[:, i]) ? 1.0 : norm(view(X, :, i))
    end
    β⃰ = [1.0 * (mod(i, Int(p / k⃰)) == 0) for i = 1:p]

    σ = sqrt(norm(X * β⃰)^2 / (n * SNR))

    y = X * β⃰ + randn(n) * σ
    yval = X * β⃰ + randn(n) * σ

    XTX = X'X

    return X, y, X, yval, XTX, p, β⃰, k⃰
end

# ============================================================================
# Function Definitions
# ============================================================================

function funcs(X, y, yval, XTX, λ₀, λ₂=0.0, λ₁=0.0)
    # T_e threshold calculation:
    # proxl0(x) with tau=1: x / (1 + 2λ₂) if |x| >= sqrt(2λ₀ * (1 + 2λ₂))
    # proxl0(x, tau): x / (1 + 2λ₂τ) if |x| >= sqrt(2λ₀τ * (1 + 2λ₂τ))
    r!(r, β) = (mul!(r, X, β); r .-= y)
    r(β) = X * β - y
    f(r) = norm(r)^2 / 2
    h(β) = λ₀ * norm(β, 0) + λ₁ * norm(β, 1) + λ₂ * norm(β)^2
    F(r, β) = f(r) + h(β)
    ∇f!(∇f, r) = mul!(∇f, X', r)
    ∇f(r) = X'r
    
    proxl1(x, thresh) = sign(x) * max(0.0, abs(x) - thresh)
    proxl0(x) = (abs(proxl1(x, λ₁)) >= sqrt(2 * λ₀ * (1 + 2λ₂))) ? proxl1(x, λ₁) / (1 + 2λ₂) : zero(x)
    proxl0(x, τ) = (abs(proxl1(x, τ * λ₁)) >= sqrt(2 * λ₀ * τ * (1 + 2 * λ₂ * τ))) ? proxl1(x, τ * λ₁) / (1 + 2 * λ₂ * τ) : zero(x)
    proxl0VM(x, Uₖ) = (abs(proxl1(x, Uₖ * λ₁)) >= sqrt(2 * λ₀ * Uₖ * (1 + 2 * λ₂ * Uₖ))) ? proxl1(x, Uₖ * λ₁) / (1 + 2 * λ₂ * Uₖ) : zero(x)

    # In-place versions for reduced allocations
    function proxl0!(out, x, τ)
        thresh_l1 = τ * λ₁
        thresh_l0 = sqrt(2λ₀ * τ * (1 + 2λ₂ * τ))
        denom = 1 + 2λ₂ * τ
        @inbounds @simd for i in eachindex(out)
            val_l1 = sign(x[i]) * max(0.0, abs(x[i]) - thresh_l1)
            out[i] = abs(val_l1) >= thresh_l0 ? val_l1 / denom : zero(eltype(x))
        end
        return out
    end

    function proxl0VM!(out, x, Uₖ)
        @inbounds @simd for i in eachindex(out)
            thresh_l1 = Uₖ[i] * λ₁
            thresh_l0 = sqrt(2λ₀ * Uₖ[i] * (1 + 2λ₂ * Uₖ[i]))
            denom = 1 + 2λ₂ * Uₖ[i]
            val_l1 = sign(x[i]) * max(0.0, abs(x[i]) - thresh_l1)
            out[i] = abs(val_l1) >= thresh_l0 ? val_l1 / denom : zero(eltype(x))
        end
        return out
    end

    return r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX
end

function find_optimal_ridge(X, y, Xval, yval; n_grid=100, l2_min=1e-4, l2_max=1e3)
    # Compute Ridge Regression path using SVD
    # (X'X + 2λI)β = X'y
    U, S, V = svd(X)
    UTy = U' * y
    
    lambdas = 10 .^ range(log10(l2_min), log10(l2_max), length=n_grid)
    best_λ2 = l2_min
    best_mse = Inf
    
    for L in lambdas
        # β = V * diag(S_i / (S_i^2 + 2L)) * U'y
        b = V * ( (S .* UTy) ./ (S.^2 .+ 2*L) )
        mse = norm(yval - Xval * b)^2
        if mse < best_mse
            best_mse = mse
            best_λ2 = L
        end
    end
    return best_λ2
end

function find_optimal_lasso(X, y, Xval, yval)
    RCall.globalEnv[:X_train_lasso] = collect(X)
    RCall.globalEnv[:y_train_lasso] = collect(y)
    try
        r_code = raw"""
        if (dir.exists("~/R_libs")) .libPaths(c("~/R_libs", .libPaths()))
        suppressMessages(library(glmnet))
        fit <- cv.glmnet(X_train_lasso, y_train_lasso, alpha=1, intercept=FALSE, standardize=FALSE)
        fit$lambda.min
        """
        return RCall.rcopy(RCall.reval(r_code))
    catch e
        @warn "Lasso cross-validation failed: $e"
        return 0.01
    end
end

# ============================================================================
# VMNSPG Algorithm
# ============================================================================

function VMNSPG(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64), µ=10^-3)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX = funcs

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

    # Initial step computation
    temp .= x⁰ .- ∇fxᵏ .* 10^-5
    r!(rᵏ, temp)
    yᵏ .= ∇fxᵏ .- ∇f(rᵏ)
    yᵏTsᵏ = dot(yᵏ, ∇fxᵏ)
    nsᵏ = γₖ¹ = dot(∇fxᵏ, ∇fxᵏ) * 10^-5 / yᵏTsᵏ
    γₖ² = yᵏTsᵏ * 10^-5 / dot(yᵏ, yᵏ)

    # Initialize Uₖ in-place
    @inbounds for i in eachindex(Uₖ)
        val = 10^-5 * ∇fxᵏ[i] * ∇fxᵏ[i] / (∇fxᵏ[i] * yᵏ[i])
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

        if 0.0 < abs(Fxᵏ⁻¹ - Fxᵏ) / Fxᵏ <= ϵ
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
# NSPGH Algorithm
# ============================================================================

function NSPGH(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64))
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX = funcs

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

    # Initial step computation
    temp .= x⁰ .- ∇fxᵏ .* 10^-5
    r!(rᵏ, temp)
    yᵏ .= ∇fxᵏ .- ∇f(rᵏ)
    yᵏTsᵏ = dot(yᵏ, ∇fxᵏ)
    nsᵏ = γₖ¹ = dot(∇fxᵏ, ∇fxᵏ) * 10^-5 / yᵏTsᵏ
    γₖ² = yᵏTsᵏ * 10^-5 / dot(yᵏ, yᵏ)
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

        if 0.0 < abs(Fxᵏ⁻¹ - Fxᵏ) / Fxᵏ <= ϵ
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
# NSPG Algorithm
# ============================================================================

function NSPG(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64), γₖ=0.0, kwargs...)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX = funcs

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
        # Initial NSPG step via BB1: γₖ = ⟨sᵏ,sᵏ⟩/⟨yᵏ,sᵏ⟩
        # sᵏ = ε∇f, yᵏ = ∇fxᵏ + X'(y - X(β - ε∇f)) = εX'X∇f
        # γₖ = ε‖∇f‖² / ⟨yᵏ, ∇f⟩ = ε‖∇f‖² / (ε⟨X'X∇f,∇f⟩)
        # ε = 10^-5
        XTX∇fxᵏ = XTX_mul(X, ∇fxᵏ)
        γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
    end
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

        if 0.0 < abs(Fxᵏ⁻¹ - Fxᵏ) / Fxᵏ <= ϵ
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
# Active Set NSPG Algorithm
# ============================================================================

function ActiveSetSPG(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64), γₖ=0.0, ActiveSetNum=6, kwargs...)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX = funcs
    
    @info "  [NSPG] Initial nnz: $(count(!iszero, x⁰))"

    # Pre-allocate all work arrays
    xᵏ = copy(x⁰)
    xᵏ⁻¹ = copy(x⁰)
    sᵏ = similar(x⁰)
    yᵏ = similar(x⁰)
    ∇fxᵏ = similar(x⁰)
    ∇fxᵏ⁻¹ = similar(x⁰)
    temp = similar(x⁰)

    rᵏ = r(xᵏ)
    y_vec = similar(rᵏ)
    mul!(y_vec, X, xᵏ)
    y_vec .-= rᵏ
    
    ∇f!(∇fxᵏ, rᵏ)
    Fxᵏ⁻¹ = Fxᵏ = F(rᵏ, xᵏ)

    if iszero(γₖ)
        XTX∇fxᵏ = XTX_mul(X, ∇fxᵏ)
        γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
    end
    nsᵏ = γₖ
    lastₘ = fill(Fxᵏ, m)

    SameSuppCounter = 0
    Stabilized = false
    S = Int[]
    total_iters = 0

    while total_iters < kₘₐₓ
        total_iters += 1

        if !Stabilized
            if has_same_support(xᵏ, xᵏ⁻¹)
                SameSuppCounter += 1
                if SameSuppCounter >= ActiveSetNum
                    S = findall(!iszero, xᵏ)
                    if !isempty(S)
                        Stabilized = true
                    end
                end
            else
                SameSuppCounter = 0
            end
        end

        Fxₗ₍ₖ₎ = maximum(lastₘ)

        if Stabilized
            # --- ACTIVE SET MODE ---
            X_S = view(X, :, S)
            xᵏ_S = view(xᵏ, S)
            xᵏ⁻¹_S = view(xᵏ⁻¹, S)
            ∇fxᵏ_S = view(∇fxᵏ, S)
            ∇fxᵏ⁻¹_S = view(∇fxᵏ⁻¹, S)
            temp_S = view(temp, S)
            sᵏ_S = view(sᵏ, S)
            yᵏ_S = view(yᵏ, S)
            
            while true
                temp_S .= xᵏ⁻¹_S .- γₖ .* ∇fxᵏ_S
                proxl0!(xᵏ_S, temp_S, γₖ)
                
                # Evaluate r and F on active set
                mul!(rᵏ, X_S, xᵏ_S)
                rᵏ .-= y_vec
                Fxᵏ = F(rᵏ, xᵏ) # xᵏ maintains 0s outside S
                
                sᵏ_S .= xᵏ_S .- xᵏ⁻¹_S
                nsᵏ = dot(sᵏ_S, sᵏ_S)

                if Fxᵏ + δ * nsᵏ / (2 * γₖ) <= Fxₗ₍ₖ₎
                    break
                end
                γₖ *= τ
                if isnan(γₖ) || γₖ < γₘᵢₙ
                    break
                end
            end

            # Check convergence within active set
            if 0.0 < abs(Fxᵏ⁻¹ - Fxᵏ) / Fxᵏ <= ϵ
                # Full gradient check
                r!(rᵏ, xᵏ)
                ∇f!(∇fxᵏ, rᵏ)
                
                temp .= xᵏ .- γₖ .* ∇fxᵏ
                proxl0!(temp, temp, γₖ) # Use temp as test array
                
                if has_same_support(temp, xᵏ)
                    return xᵏ, total_iters
                else
                    # Break stabilization, support wants to change
                    Stabilized = false
                    SameSuppCounter = 0
                    
                    # Restart BB calculation for full vector
                    XTX∇fxᵏ = XTX_mul(X, ∇fxᵏ)
                    γ_denom = dot(XTX∇fxᵏ, ∇fxᵏ)
                    if γ_denom > 0
                        γₖ = dot(∇fxᵏ, ∇fxᵏ) / γ_denom
                    end
                    
                    copyto!(xᵏ⁻¹, xᵏ)
                    copyto!(∇fxᵏ⁻¹, ∇fxᵏ)
                    Fxᵏ⁻¹ = Fxᵏ
                    popfirst!(lastₘ)
                    push!(lastₘ, Fxᵏ)
                    continue
                end
            end

            popfirst!(lastₘ)
            push!(lastₘ, Fxᵏ)
            copyto!(xᵏ⁻¹_S, xᵏ_S)
            Fxᵏ⁻¹ = Fxᵏ
            copyto!(∇fxᵏ⁻¹_S, ∇fxᵏ_S)
            
            mul!(∇fxᵏ_S, X_S', rᵏ)
            yᵏ_S .= ∇fxᵏ_S .- ∇fxᵏ⁻¹_S
            
            yᵏTsᵏ = dot(yᵏ_S, sᵏ_S)
            γₖ = yᵏTsᵏ > 0 ? nsᵏ / yᵏTsᵏ : sqrt(nsᵏ / dot(yᵏ_S, yᵏ_S))
            if γₖ > γₘₐₓ || γₖ < γₘᵢₙ
                γₖ = sqrt(nsᵏ / dot(yᵏ_S, yᵏ_S))
            end

        else
            # --- FULL SWEEP MODE ---
            while true
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

            if 0.0 < abs(Fxᵏ⁻¹ - Fxᵏ) / Fxᵏ <= ϵ
                return xᵏ, total_iters
            end

            popfirst!(lastₘ)
            push!(lastₘ, Fxᵏ)
            copyto!(xᵏ⁻¹, xᵏ)
            Fxᵏ⁻¹ = Fxᵏ
            copyto!(∇fxᵏ⁻¹, ∇fxᵏ)
            ∇f!(∇fxᵏ, rᵏ)

            yᵏ .= ∇fxᵏ .- ∇fxᵏ⁻¹
            yᵏTsᵏ = dot(yᵏ, sᵏ)
            γₖ = yᵏTsᵏ > 0 ? nsᵏ / yᵏTsᵏ : sqrt(nsᵏ / dot(yᵏ, yᵏ))
            if γₖ > γₘₐₓ || γₖ < γₘᵢₙ
                γₖ = sqrt(nsᵏ / dot(yᵏ, yᵏ))
            end
        end
    end

    return xᵏ, kₘₐₓ
end

# ============================================================================
# RCall Integration for L0Learn
# ============================================================================

const R_SOLVE_SCRIPT_REF = Ref{Any}(nothing)

function get_r_solve_script()
    if R_SOLVE_SCRIPT_REF[] === nothing
        R_SOLVE_SCRIPT_REF[] = RCall.rparse(raw"""
            if (dir.exists("~/R_libs")) {
                .libPaths(c("~/R_libs", .libPaths()))
            }
            suppressMessages(library(L0Learn))
            
            n <- nrow(X_train)
            lambda_scaled <- lambda_val / n
            penalty_str <- if (l2_val > 0.0) "L0L2" else if (l1_val > 0.0) "L0L1" else "L0"
            
            if (penalty_str == "L0L2") {
                gamma_R <- l2_val / (2.0 * lambda_val)
                if (use_warmstart_val) {
                    fit <- L0Learn.fit(X_train, y_train, penalty=penalty_str, lambdaGrid=list(c(lambda_scaled)), nGamma=1, gammaMax=gamma_R, gammaMin=gamma_R, maxSuppSize=maxSuppSize, intercept=FALSE, initialBeta=beta_init)
                } else {
                    fit <- L0Learn.fit(X_train, y_train, penalty=penalty_str, lambdaGrid=list(c(lambda_scaled)), nGamma=1, gammaMax=gamma_R, gammaMin=gamma_R, maxSuppSize=maxSuppSize, intercept=FALSE)
                }
            } else if (penalty_str == "L0L1") {
                gamma_R <- l1_val / lambda_val
                if (use_warmstart_val) {
                    fit <- L0Learn.fit(X_train, y_train, penalty=penalty_str, lambdaGrid=list(c(lambda_scaled)), nGamma=1, gammaMax=gamma_R, gammaMin=gamma_R, maxSuppSize=maxSuppSize, intercept=FALSE, initialBeta=beta_init)
                } else {
                    fit <- L0Learn.fit(X_train, y_train, penalty=penalty_str, lambdaGrid=list(c(lambda_scaled)), nGamma=1, gammaMax=gamma_R, gammaMin=gamma_R, maxSuppSize=maxSuppSize, intercept=FALSE)
                }
            } else {
                if (use_warmstart_val) {
                    fit <- L0Learn.fit(X_train, y_train, penalty=penalty_str, lambdaGrid=list(c(lambda_scaled)), maxSuppSize=maxSuppSize, intercept=FALSE, initialBeta=beta_init)
                } else {
                    fit <- L0Learn.fit(X_train, y_train, penalty=penalty_str, lambdaGrid=list(c(lambda_scaled)), maxSuppSize=maxSuppSize, intercept=FALSE)
                }
            }
            
            if (length(fit$lambda[[1]]) > 0) {
                # Get coefficients for the first lambda/gamma
                gamma_val <- if (penalty_str != "L0") {
                    if (penalty_str == "L0L2") l2_val else l1_val
                } else {
                    0
                }
                beta <- coef(fit, lambda=fit$lambda[[1]][1], gamma=gamma_val)
                beta_vec <- as.numeric(beta)
            } else {
                beta_vec <- rep(0, ncol(X_train))
            }
            beta_vec
        """)
    end
    return R_SOLVE_SCRIPT_REF[]
end

function solve_l0_single_lambda_R(lambda_val, l2_val, l1_val, x_init; maxSuppSize=length(x_init), warmstart=use_warmstart)
    try
        RCall.globalEnv[:lambda_val] = lambda_val
        RCall.globalEnv[:l2_val] = l2_val
        RCall.globalEnv[:l1_val] = l1_val
        RCall.globalEnv[:maxSuppSize] = maxSuppSize
        RCall.globalEnv[:use_warmstart_val] = warmstart
        if warmstart
            RCall.globalEnv[:beta_init] = x_init
        end
        return RCall.rcopy(RCall.reval(get_r_solve_script()))
    catch e
        @warn "L0Learn R solve failed: $e"
        return zeros(length(x_init))
    end
end

function initialize_l0learn_R(vars)
    X, y, Xval, yval, XTX, p, β⃰, k⃰ = vars
    RCall.globalEnv[:X_train] = collect(X)
    RCall.globalEnv[:y_train] = collect(y)
    RCall.globalEnv[:X_val] = collect(Xval)
    RCall.globalEnv[:y_val] = collect(yval)
end

function L0LearnStep(xᵏ, funcs; lambda_val=nothing, l2_val=0.0, l1_val=0.0, X_data=nothing, y_data=nothing, warmstart=use_warmstart, kwargs...)
    if lambda_val === nothing
        return xᵏ, 0
    end
    
    beta_new = solve_l0_single_lambda_R(lambda_val, l2_val, l1_val, xᵏ; maxSuppSize=length(xᵏ), warmstart=warmstart)
    return beta_new, 1
end

function pure_l0learn_solver_R(vars; maxSuppSize=size(vars[1], 2), l2_val=0.0, l1_val=0.0)
    X, y, Xval, yval, XTX, p, β⃰, k⃰ = vars
    RCall.globalEnv[:X_train] = X
    RCall.globalEnv[:y_train] = y
    RCall.globalEnv[:X_val] = Xval
    RCall.globalEnv[:y_val] = yval
    RCall.globalEnv[:maxSuppSize] = maxSuppSize
    RCall.globalEnv[:fixed_l2] = l2_val
    RCall.globalEnv[:fixed_l1] = l1_val

    try
        r_code = raw"""
        if (dir.exists("~/R_libs")) .libPaths(c("~/R_libs", .libPaths()))
        suppressMessages(library(L0Learn))
        penalty_str <- if (fixed_l2 > 0.0) "L0L2" else if (fixed_l1 > 0.0) "L0L1" else "L0"
        
        if (penalty_str == "L0L2") {
            fit <- L0Learn.fit(X_train, y_train, penalty=penalty_str, maxSuppSize=maxSuppSize, intercept=FALSE, nGamma=1, gammaMax=fixed_l2, gammaMin=fixed_l2)
        } else if (penalty_str == "L0L1") {
            fit <- L0Learn.fit(X_train, y_train, penalty=penalty_str, maxSuppSize=maxSuppSize, intercept=FALSE, nGamma=1, gammaMax=fixed_l1, gammaMin=fixed_l1)
        } else {
            fit <- L0Learn.fit(X_train, y_train, penalty=penalty_str, maxSuppSize=maxSuppSize, intercept=FALSE)
        }
        
        best_mse <- Inf
        best_coef <- rep(0, ncol(X_train))
        best_lambda <- 0.0
        
        # Cross-validate over the path
        lambdas <- fit$lambda[[1]]
        for (i in 1:length(lambdas)) {
            beta <- as.numeric(coef(fit, lambda=lambdas[i], gamma=if(penalty_str != "L0") (if(penalty_str == "L0L2") fixed_l2 else fixed_l1) else 0))
            mse <- mean((y_val - as.numeric(X_val %*% beta))^2)
            if (mse < best_mse) {
                best_mse <- mse
                best_coef <- beta
                best_lambda <- lambdas[i]
            }
        }
        best_coef_r <- best_coef
        best_lambda_r <- best_lambda
        """
        RCall.reval(r_code)
        best_coef = Float64.(RCall.rcopy(RCall.reval("best_coef_r")))
        best_lambda = Float64(RCall.rcopy(RCall.reval("best_lambda_r")))
        @info "  Best Lambda 0 (L0Learn Native): $best_lambda"
        return best_coef
    catch e
        @warn "Pure L0Learn solver failed: $e"
        return zeros(p)
    end
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
# NSPG + PGCCD Combined
# ============================================================================

function PGCCD(x⁰, funcs; sortperc=1 / 4, ActiveSetNum=6, work_xᵏ=nothing, work_rᵏ=nothing, work_xᵏ⁻¹=nothing, work_∇f=nothing, work_order=nothing, kwargs...)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX = funcs
    @info "  [PGCCD] Initial nnz: $(count(!iszero, x⁰))"

    if work_xᵏ !== nothing
        xᵏ = work_xᵏ
        copyto!(xᵏ, x⁰)
        rᵏ = work_rᵏ
        r!(rᵏ, xᵏ)
        rᵏ .= .-rᵏ
        xᵏ⁻¹ = work_xᵏ⁻¹
        copyto!(xᵏ⁻¹, xᵏ)
        n_vars = length(x⁰)
        ksort = round(Int64, n_vars * sortperc)
        ∇fxᵏ = work_∇f
        ∇f!(∇fxᵏ, rᵏ)
        if work_order !== nothing
            for i in 1:n_vars
                work_order[i] = i
            end
            sort!(work_order, by = i -> abs(∇fxᵏ[i]), rev=true)
            greedy = view(work_order, 1:n_vars)
        else
            greedy_idx = partialsortperm(abs.(∇fxᵏ), 1:ksort, rev=true)
            greedy = vcat(greedy_idx, setdiff(1:n_vars, greedy_idx))
        end
    else
        xᵏ = copy(x⁰)
        rᵏ = -r(xᵏ)
        xᵏ⁻¹ = copy(xᵏ)
        n_vars = length(x⁰)
        ksort = round(Int64, n_vars * sortperc)
        greedy = partialsortperm(abs.(∇f(rᵏ)), 1:ksort, rev=true)
        greedy = vcat(greedy, setdiff(1:n_vars, greedy))
    end
    Fxᵏ⁻¹ = F(rᵏ, xᵏ)
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
                    if work_order !== nothing
                        # DO NOT MUTATE work_order! It destroys the greedy order!
                        # We allocate a new array or just findall here, or better yet,
                        # since we want to avoid allocation, we can mutate the END of work_order?
                        # No, just allocate a small vector for the active set view,
                        # it's tiny (k elements). The allocation overhead is negligible
                        # compared to corrupting the greedy order.
                        Order = findall(!iszero, xᵏ)
                    else
                        Order = findall(!iszero, xᵏ)
                    end
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
            if Stabilized
                # Final optimality check (full sweep) to guarantee CW-minimum
                optimality_violated = false
                @inbounds for i in 1:n_vars
                    # Check all variables, not just the active set
                    xi = proxl0(dot(rᵏ, view(X, :, i)) + xᵏ[i])
                    if xi != xᵏ[i]
                        BLAS.axpy!(xᵏ[i] - xi, view(X, :, i), rᵏ)
                        xᵏ[i] = xi
                        optimality_violated = true
                    end
                end
                
                if !optimality_violated
                    return xᵏ, k
                else
                    # If optimality violated, return to full sweep mode
                    Stabilized = false
                    SameSuppCounter = 0
                    Order = greedy
                    Fxᵏ⁻¹ = F(rᵏ, xᵏ)
                    continue
                end
            else
                return xᵏ, k
            end
        end

        Fxᵏ⁻¹ = Fxᵏ
    end

    return xᵏ, kₘₐₓ
end

function SPGpL0Learn(xᵏ, funcs; lambda_val=nothing, l2_val=0.0, l1_val=0.0, X_data=nothing, y_data=nothing, work_xᵏ=nothing, kwargs...)
    # Use work_xᵏ if provided, otherwise standard
    x = work_xᵏ !== nothing ? copyto!(work_xᵏ, xᵏ) : copy(xᵏ)
    x, k = NSPG(x, funcs; kwargs...)
    x, k2 = L0LearnStep(x, funcs; lambda_val=lambda_val, l2_val=l2_val, l1_val=l1_val, X_data=X_data, y_data=y_data, kwargs...)
    return x, k + k2
end

function NSPG_PGCCD(x⁰, funcs; γₖ=0.0, work_xᵏ=nothing, work_rᵏ=nothing, work_xᵏ⁻¹=nothing, work_∇f=nothing, lambda_val=nothing, X_data=nothing, y_data=nothing, kwargs...)
    x, k = NSPG(x⁰, funcs; γₖ=γₖ, kwargs...)
    x, k2 = PGCCD(x, funcs; kwargs...)
    return x, k + k2
end

# ============================================================================
# PSI1 Algorithm
# ============================================================================

function PSI1(xˡ, funcs; work_xˡ=nothing, work_sˡ=nothing, kwargs...)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX = funcs

    if work_xˡ !== nothing
        copyto!(work_xˡ, xˡ)
        xˡ = work_xˡ
    end

    # Cache index arrays to avoid repeated allocation
    nonzero_indices = findall(!iszero, xˡ)
    zero_indices = findall(iszero, xˡ)

    r⃰ = -X'r(xˡ)

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
# Solver with PSI1
# ============================================================================

function SolverPSI1(solver, x⁰, funcs; γₖ=0.0, max_psi1_fails=10, work_xˡ=nothing, work_sˡ=nothing, kwargs...)
    β = x⁰
    kᵢ = kₒ = 0
    isPSI1 = false

    # Cycle detection: track support patterns
    seen_supports = Set{UInt64}()
    consecutive_fails = 0

    while !isPSI1 && kₒ < kₘₐₓ
        kₒ += 1
        β, k = solver(β, funcs; γₖ=γₖ, kwargs..., work_xˡ=work_xˡ, work_sˡ=work_sˡ)
        kᵢ += k
        γₖ = 0.0 # Only the first call uses the pre-computed validation γₖ
        β, isPSI1 = PSI1(β, funcs; work_xˡ=work_xˡ, work_sˡ=work_sˡ, kwargs...)

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

    return copy(β), kᵢ, kₒ
end

# ============================================================================
# Cross Validation
# ============================================================================

function cross_validation(solver, vars; λ_min_ratio=10^-5, NSPG=false, stagnation_handling=true, lambda_val=nothing, use_refinement=false, fixed_l2=0.0, fixed_l1=0.0)
    X, y, Xval, yval, XTX, p, β⃰, k⃰ = vars
    suppsim(β) = count(i -> !iszero(β⃰[i]) && !iszero(β[i]), 1:p) / max(k⃰, norm(β, 0))
    predval(β) = norm(Xval * β .- yval)^2 / norm(yval)^2

    β = β_best = zeros(p)
    best = best_λ = Inf
    
    work_xᵏ = zeros(p)
    work_xᵏ⁻¹ = zeros(p)
    work_rᵏ = zeros(size(X, 1))
    work_∇f = zeros(p)
    work_xˡ = zeros(p)
    work_sˡ = zeros(p)
    work_order = collect(1:p)

    ∇fxᵏ = -X'y

    if NSPG
        XTX∇fxᵏ = XTX_mul(X, ∇fxᵏ)
        denom = dot(XTX∇fxᵏ, ∇fxᵏ)
        γₖ = denom > 1e-12 ? dot(∇fxᵏ, ∇fxᵏ) / denom : 1.0
    else
        γₖ = 1.0
    end

    # λ formula uses γₖ, fixed_l1, and fixed_l2
    max_corr = ThreadsX.maximum(abs(∇fxᵏ[j]) for j = 1:p)
    λ = 1.01 * γₖ * max(0.0, max_corr - fixed_l1)^2 / (2 * (1 + 2 * fixed_l2 * γₖ))
    λ = max(1e-8, λ)
    λ_min = λ * λ_min_ratio

    i = 1
    stagnant_count = 0
    prev_computed_λ = λ
    λ_path = Float64[]

    while λ > λ_min && norm(β, 0) != p
        push!(λ_path, λ)
        β, kᵢ, kₒ = solver(β, funcs(X, y, yval, XTX, λ, fixed_l2, fixed_l1); γₖ=NSPG ? γₖ : 0.0, work_xᵏ=work_xᵏ, work_xᵏ⁻¹=work_xᵏ⁻¹, work_rᵏ=work_rᵏ, work_∇f=work_∇f, work_xˡ=work_xˡ, work_sˡ=work_sˡ, work_order=work_order)

        mse = norm(yval .- Xval * β)
        if mse < best
            β_best = copy(β)
            best = mse
            best_λ = λ
        end

        if norm(β, 0) == p
            break
        end

        r_vec = X * β - y
        if NSPG
            ∇fxᵏ = X' * r_vec
            XTX∇fxᵏ = XTX_mul(X, ∇fxᵏ)
            denom = dot(XTX∇fxᵏ, ∇fxᵏ)
            γₖ = denom > 1e-12 ? dot(∇fxᵏ, ∇fxᵏ) / denom : 1.0
        end

        # Compute new λ from formula - with min to previous λ for consistency
        # The 0.9 multiplier is OUTSIDE the min for monotonic decrease
        if norm(β, 0) != p
            max_r = ThreadsX.maximum(abs(dot(view(X, :, j), r_vec)) for j = 1:p if iszero(β[j]))
            raw_λ = γₖ * max(0.0, max_r - fixed_l1)^2 / (2 * (1 + 2 * fixed_l2 * γₖ))
        else
            raw_λ = 0.0
        end
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
    β_best, kᵢ, kₒ = solver(β_best, funcs(X, y, yval, XTX, best_λ, fixed_l2, fixed_l1); γₖ=0.0, lambda_val=best_λ, l2_val=fixed_l2, l1_val=fixed_l1, X_data=X, y_data=y, work_xᵏ=work_xᵏ, work_xᵏ⁻¹=work_xᵏ⁻¹, work_rᵏ=work_rᵏ, work_∇f=work_∇f, work_xˡ=work_xˡ, work_sˡ=work_sˡ, work_order=work_order)
    if use_refinement
        β_best_0, kᵢ, kₒ = solver(zeros(p), funcs(X, y, yval, XTX, best_λ, fixed_l2, fixed_l1); γₖ=0.0, lambda_val=best_λ, l2_val=fixed_l2, l1_val=fixed_l1, X_data=X, y_data=y, work_xᵏ=work_xᵏ, work_xᵏ⁻¹=work_xᵏ⁻¹, work_rᵏ=work_rᵏ, work_∇f=work_∇f, work_xˡ=work_xˡ, work_sˡ=work_sˡ, work_order=work_order)
        zero_won = predval(β_best_0) < predval(β_best)
        if zero_won
            si = suppsim(β_best_0) - suppsim(β_best)
            ss = abs(si) < 1e-6 ? 0 : (si > 0 ? 1 : -1)
        else
            si = 0.0
            ss = -9 # Rejected
        end
        β_best = zero_won ? β_best_0 : β_best
        @info "Forward CV Path" start_λ=(isempty(λ_path) ? λ : λ_path[1]) end_λ=(isempty(λ_path) ? λ : λ_path[end]) steps=length(λ_path) max_support=norm(β_best, 0)
        return β_best, best_λ, norm(β_best, 0), predval(β_best), suppsim(β_best), norm(β_best - β⃰, Inf), zero_won, ss, si
    else
        @info "Forward CV Path" start_λ=(isempty(λ_path) ? λ : λ_path[1]) end_λ=(isempty(λ_path) ? λ : λ_path[end]) steps=length(λ_path) max_support=norm(β_best, 0)
        return β_best, best_λ, norm(β_best, 0), predval(β_best), suppsim(β_best), norm(β_best - β⃰, Inf), false, 0, 0.0
    end
end

# ============================================================================
# Inverse Cross Validation (lambda grows instead of shrinks)
# ============================================================================



function inverse_cross_validation(solver, vars; λ_max_ratio=1e30, NSPG=false, stagnation_handling=true, use_refinement=false, fixed_l2=0.0, fixed_l1=0.0)
    X, y, Xval, yval, XTX, p, β⃰, k⃰ = vars
    suppsim(β) = count(i -> !iszero(β⃰[i]) && !iszero(β[i]), 1:p) / max(k⃰, norm(β, 0))
    predval(β) = norm(Xval * β .- yval)^2 / norm(yval)^2

    β = β_best = zeros(p)
    best = best_λ = Inf

    # Compute initial residual correlations (residual = y when β = 0)
    work_xᵏ = zeros(p)
    work_xᵏ⁻¹ = zeros(p)
    work_rᵏ = zeros(size(X, 1))
    work_∇f = zeros(p)
    work_xˡ = zeros(p)
    work_sˡ = zeros(p)
    work_order = collect(1:p)

    ∇fxᵏ = -X'y

    # Initial λ: use min_corr for inverse CV (we start low and go high)
    min_corr = ThreadsX.minimum(abs(∇fxᵏ[j]) for j = 1:p)

    if NSPG
        XTX∇fxᵏ = XTX_mul(X, ∇fxᵏ)
        γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
    else
        γₖ = 1.0
    end

    # Initial λ for inverse CV: start SMALL to get high support, then increase
    λ = max(1e-8, (1.01^-1) * γₖ * max(0.0, min_corr - fixed_l1)^2 / (2 * (1 + 2 * fixed_l2 * γₖ)))

    # --- PRE-LOOP WARMUP: Reduce λ until we get non-zero support ---
    # For L0Learn with const correlation, the initial λ may be too large.
    β_warmup, _, _ = solver(zeros(p), funcs(X, y, yval, XTX, λ, fixed_l2, fixed_l1); γₖ=NSPG ? γₖ : 0.0, work_xᵏ=work_xᵏ, work_xᵏ⁻¹=work_xᵏ⁻¹, work_rᵏ=work_rᵏ, work_∇f=work_∇f, work_xˡ=work_xˡ, work_sˡ=work_sˡ, work_order=work_order)
    while iszero(norm(β_warmup, 0))
        λ *= 0.9
        β_warmup, _, _ = solver(zeros(p), funcs(X, y, yval, XTX, λ, fixed_l2, fixed_l1); γₖ=NSPG ? γₖ : 0.0, work_xᵏ=work_xᵏ, work_xᵏ⁻¹=work_xᵏ⁻¹, work_rᵏ=work_rᵏ, work_∇f=work_∇f, work_xˡ=work_xˡ, work_sˡ=work_sˡ, work_order=work_order)
    end
    β = β_warmup
    best = norm(yval .- Xval * β)
    β_best = copy(β)
    best_λ = λ
    # --- END WARMUP ---

    λ_max = λ * λ_max_ratio

    i = 1
    stagnant_count = 0
    prev_λ = λ

    while λ < λ_max && (!iszero(norm(β, 0)) || i == 1)
        # Ensure λ stays positive
        λ = max(λ, eps())

        # Run solver
        β, kᵢ, kₒ = solver(β, funcs(X, y, yval, XTX, λ, fixed_l2, fixed_l1); γₖ=NSPG ? γₖ : 0.0, work_xᵏ=work_xᵏ, work_xᵏ⁻¹=work_xᵏ⁻¹, work_rᵏ=work_rᵏ, work_∇f=work_∇f, work_xˡ=work_xˡ, work_sˡ=work_sˡ, work_order=work_order)

        mse = norm(yval .- Xval * β)
        if mse < best
            β_best = copy(β)
            best = mse
            best_λ = λ
        end

        # Stop if support is zero
        if norm(β, 0) == 0
            break
        end

        r_vec = X * β - y
        if NSPG
            ∇fxᵏ = X' * r_vec
            XTX∇fxᵏ = XTX_mul(X, ∇fxᵏ)
            denom = dot(XTX∇fxᵏ, ∇fxᵏ)
            γₖ = denom > 1e-12 ? dot(∇fxᵏ, ∇fxᵏ) / denom : 1.0
        end

        # Compute new λ from formula - with max to previous λ for monotonic increase
        # The 0.9^-1 multiplier is OUTSIDE the max
        raw_λ = ThreadsX.minimum(
            max(0.0, abs(β[j] - γₖ * dot(view(X, :, j), r_vec)) - γₖ * fixed_l1)^2
            for j = 1:p if !iszero(β[j])
        ) / (2 * γₖ * (1 + 2 * fixed_l2 * γₖ))
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
    β_best, kᵢ, kₒ = solver(β_best, funcs(X, y, yval, XTX, best_λ, fixed_l2, fixed_l1); γₖ=0.0, lambda_val=best_λ, l2_val=fixed_l2, l1_val=fixed_l1, X_data=X, y_data=y, work_xᵏ=work_xᵏ, work_xᵏ⁻¹=work_xᵏ⁻¹, work_rᵏ=work_rᵏ, work_∇f=work_∇f, work_xˡ=work_xˡ, work_sˡ=work_sˡ, work_order=work_order)
    if use_refinement
        β_best_0, kᵢ, kₒ = solver(zeros(p), funcs(X, y, yval, XTX, best_λ, fixed_l2, fixed_l1); γₖ=0.0, lambda_val=best_λ, l2_val=fixed_l2, l1_val=fixed_l1, X_data=X, y_data=y, work_xᵏ=work_xᵏ, work_xᵏ⁻¹=work_xᵏ⁻¹, work_rᵏ=work_rᵏ, work_∇f=work_∇f, work_xˡ=work_xˡ, work_sˡ=work_sˡ, work_order=work_order)
        zero_won = predval(β_best_0) < predval(β_best)
        if zero_won
            si = suppsim(β_best_0) - suppsim(β_best)
            ss = abs(si) < 1e-6 ? 0 : (si > 0 ? 1 : -1)
        else
            si = 0.0
            ss = -9 # Rejected
        end
        β_best = zero_won ? β_best_0 : β_best
        return β_best, best_λ, norm(β_best, 0), predval(β_best), suppsim(β_best), norm(β_best - β⃰, Inf), zero_won, ss, si
    else
        return β_best, best_λ, norm(β_best, 0), predval(β_best), suppsim(β_best), norm(β_best - β⃰, Inf), false, 0, 0.0
    end
end
function smart_adaptive_cross_validation(solver, vars;
    λ_min_ratio=10^-5,
    λ_max_ratio=floatmax(),
    NSPG=false,
    stagnation_handling=true,
    lambda_val=nothing,
    use_refinement=false, fixed_l2=0.0, fixed_l1=0.0)

    X, y, Xval, yval, XTX, p, β⃰, k⃰ = vars
    suppsim(β) = count(i -> !iszero(β⃰[i]) && !iszero(β[i]), 1:p) / max(k⃰, norm(β, 0))
    predval(β) = norm(Xval * β .- yval)^2 / norm(yval)^2
    calc_mse(β) = norm(yval .- Xval * β)

    # --- 1. PROBE PHASE ---
    work_xᵏ = zeros(p)
    work_xᵏ⁻¹ = zeros(p)
    work_rᵏ = zeros(size(X, 1))
    work_∇f = zeros(p)
    work_xˡ = zeros(p)
    work_sˡ = zeros(p)
    work_order = collect(1:p)
    
    ∇fxᵏ = -X'y
    correlations = abs.(∇fxᵏ)
    max_corr = ThreadsX.maximum(correlations)
    min_corr = ThreadsX.minimum(correlations)

    if NSPG
        XTX∇fxᵏ = XTX_mul(X, ∇fxᵏ)
        γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
    else
        γₖ = 1.0
    end

    λ_low = max(1e-8, (1.01^-1) * γₖ * max(0.0, min_corr - fixed_l1)^2 / (2 * (1 + 2 * fixed_l2 * γₖ)))
    λ_high = max(1e-8, 1.01 * γₖ * max(0.0, max_corr - fixed_l1)^2 / (2 * (1 + 2 * fixed_l2 * γₖ)))

    # Probe Low
    β_low, _, _ = solver(zeros(p), funcs(X, y, yval, XTX, λ_low, fixed_l2, fixed_l1); γₖ=NSPG ? γₖ : 0.0, lambda_val=λ_low, l2_val=fixed_l2, l1_val=fixed_l1, X_data=X, y_data=y, work_xᵏ=work_xᵏ, work_xᵏ⁻¹=work_xᵏ⁻¹, work_rᵏ=work_rᵏ, work_∇f=work_∇f, work_xˡ=work_xˡ, work_sˡ=work_sˡ, work_order=work_order)
    
    # --- PRE-LOOP WARMUP: Reduce λ_low until we get non-zero support ---
    # For L0Learn with const correlation, λ_low may still be too large.
    while iszero(norm(β_low, 0))
        λ_low *= 0.9
        β_low, _, _ = solver(zeros(p), funcs(X, y, yval, XTX, λ_low, fixed_l2, fixed_l1); γₖ=NSPG ? γₖ : 0.0, lambda_val=λ_low, l2_val=fixed_l2, l1_val=fixed_l1, X_data=X, y_data=y, work_xᵏ=work_xᵏ, work_xᵏ⁻¹=work_xᵏ⁻¹, work_rᵏ=work_rᵏ, work_∇f=work_∇f, work_xˡ=work_xˡ, work_sˡ=work_sˡ, work_order=work_order)
    end
    # --- END WARMUP ---
    
    mse_low = calc_mse(β_low)

    # Probe High
    β_high, _, _ = solver(zeros(p), funcs(X, y, yval, XTX, λ_high, fixed_l2, fixed_l1); γₖ=NSPG ? γₖ : 0.0, lambda_val=λ_high, l2_val=fixed_l2, l1_val=fixed_l1, X_data=X, y_data=y, work_xᵏ=work_xᵏ, work_xᵏ⁻¹=work_xᵏ⁻¹, work_rᵏ=work_rᵏ, work_∇f=work_∇f, work_xˡ=work_xˡ, work_sˡ=work_sˡ, work_order=work_order)
    mse_high = calc_mse(β_high)

    # Select Best Start
    if mse_low < mse_high
        current_λ = λ_low
        current_β = copy(β_low)
        current_mse = mse_low
        direction = :increase
        λ_limit = λ_high * λ_max_ratio
    else
        current_λ = λ_high
        current_β = copy(β_high)
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
            raw_λ = SUP == 0 ? 0.0 : ThreadsX.minimum(
                max(0.0, abs(current_β[j] - γₖ * dot(view(X, :, j), rᵏ)) - γₖ * fixed_l1)^2
                for j = 1:p if !iszero(current_β[j])
            ) / (2 * γₖ * (1 + 2 * fixed_l2 * γₖ))
            # max with previous λ, THEN multiply by 0.9^-1
            computed_λ = (0.9^-1) * max(current_λ, raw_λ)
        else # :decrease
            SUP = Int(norm(current_β, 0))
            rᵏ = X * current_β - y
            val = SUP == p ? 0.0 : ThreadsX.maximum(abs(dot(view(X, :, j), rᵏ)) for j = 1:p if iszero(current_β[j]))
            raw_λ = γₖ * max(0.0, val - fixed_l1)^2 / (2 * (1 + 2 * fixed_l2 * γₖ))
            # just multiply by 0.9 (fix restored to trigger the explosion for testing)
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
        probe_β, _, _ = solver(current_β, funcs(X, y, yval, XTX, probe_λ, fixed_l2, fixed_l1); γₖ=NSPG ? γₖ : 0.0, lambda_val=probe_λ, l2_val=fixed_l2, l1_val=fixed_l1, X_data=X, y_data=y, work_xᵏ=work_xᵏ, work_xᵏ⁻¹=work_xᵏ⁻¹, work_rᵏ=work_rᵏ, work_∇f=work_∇f, work_xˡ=work_xˡ, work_sˡ=work_sˡ, work_order=work_order)
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
                raw_λ = SUP == 0 ? 0.0 : ThreadsX.minimum(
                    max(0.0, abs(current_β[j] - γₖ * dot(view(X, :, j), rᵏ)) - γₖ * fixed_l1)^2
                    for j = 1:p if !iszero(current_β[j])
                ) / (2 * γₖ * (1 + 2 * fixed_l2 * γₖ))
                alt_λ = (0.9^-1) * max(current_λ, raw_λ)
            else # :decrease
                SUP = Int(norm(current_β, 0))
                rᵏ = X * current_β - y
                val = SUP == p ? 0.0 : ThreadsX.maximum(abs(dot(view(X, :, j), rᵏ)) for j = 1:p if iszero(current_β[j]))
                raw_λ = γₖ * max(0.0, val - fixed_l1)^2 / (2 * (1 + 2 * fixed_l2 * γₖ))
                alt_λ = 0.9 * min(current_λ, raw_λ)
            end

            alt_β, _, _ = solver(current_β, funcs(X, y, yval, XTX, alt_λ, fixed_l2, fixed_l1); γₖ=NSPG ? γₖ : 0.0, lambda_val=alt_λ, l2_val=fixed_l2, l1_val=fixed_l1, X_data=X, y_data=y, work_xᵏ=work_xᵏ, work_xᵏ⁻¹=work_xᵏ⁻¹, work_rᵏ=work_rᵏ, work_∇f=work_∇f, work_xˡ=work_xˡ, work_sˡ=work_sˡ, work_order=work_order)
            alt_mse = calc_mse(alt_β)
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

        # Update γₖ if NSPG mode
        if NSPG
            ∇fxᵏ = X' * (X * current_β - y)
            XTX∇fxᵏ = XTX_mul(X, ∇fxᵏ)
            denom = dot(XTX∇fxᵏ, ∇fxᵏ)
            γₖ = denom > 1e-12 ? dot(∇fxᵏ, ∇fxᵏ) / denom : 1.0
        end
        i += 1
    end

    if !use_refinement
        return best_β, best_λ, norm(best_β, 0), predval(best_β), suppsim(best_β), norm(best_β - β⃰, Inf), false, 0, 0.0
    end

    # Final refinement with best λ
    β_best_path, kᵢ, kₒ = solver(best_β, funcs(X, y, yval, XTX, best_λ, fixed_l2, fixed_l1); γₖ=0.0, lambda_val=best_λ, l2_val=fixed_l2, l1_val=fixed_l1, X_data=X, y_data=y, work_xᵏ=work_xᵏ, work_xᵏ⁻¹=work_xᵏ⁻¹, work_rᵏ=work_rᵏ, work_∇f=work_∇f, work_xˡ=work_xˡ, work_sˡ=work_sˡ, work_order=work_order)
    β_best_zero, kᵢ, kₒ = solver(zeros(p), funcs(X, y, yval, XTX, best_λ, fixed_l2, fixed_l1); γₖ=0.0, lambda_val=best_λ, l2_val=fixed_l2, l1_val=fixed_l1, X_data=X, y_data=y, work_xᵏ=work_xᵏ, work_xᵏ⁻¹=work_xᵏ⁻¹, work_rᵏ=work_rᵏ, work_∇f=work_∇f, work_xˡ=work_xˡ, work_sˡ=work_sˡ, work_order=work_order)
    
    zero_won = predval(β_best_zero) < predval(β_best_path)
    if zero_won
        si = suppsim(β_best_zero) - suppsim(β_best_path)
        ss = abs(si) < 1e-6 ? 0 : (si > 0 ? 1 : -1)
    else
        si = 0.0
        ss = -9 # Rejected
    end
    
    β_ref = zero_won ? β_best_zero : β_best_path

    return β_ref, best_λ, norm(β_ref, 0), predval(β_ref), suppsim(β_ref), norm(β_ref - β⃰, Inf), zero_won, ss, si
end
function outer_l2_smart_cross_validation(solver, vars; inner_cv=smart_adaptive_cross_validation, kw...)
    X, y, Xval, yval, XTX, p, β⃰, k⃰ = vars
    opt_ridge = find_optimal_ridge(X, y, Xval, yval)
    lambdas2 = 10 .^ range(log10(max(1e-6, 0.1*opt_ridge)), log10(10*opt_ridge), length=15)
    n_grid = length(lambdas2)
    
    # 1. Test Low Extreme
    l2_low = lambdas2[1]
    res_low = inner_cv(solver, vars; fixed_l2=l2_low, kw...)
    mse_low = norm(yval .- Xval * res_low[1])^2
    
    # 2. Test High Extreme
    l2_high = lambdas2[end]
    res_high = inner_cv(solver, vars; fixed_l2=l2_high, kw...)
    mse_high = norm(yval .- Xval * res_high[1])^2
    
    if mse_low < mse_high
        best_res = res_low
        best_mse = mse_low
        best_idx = 1
        direction = 1
    else
        best_res = res_high
        best_mse = mse_high
        best_idx = n_grid
        direction = -1
    end
    
    # 3. Move inwards until MSE worsens
    curr_idx = best_idx + direction
    while 1 <= curr_idx <= n_grid
        l2 = lambdas2[curr_idx]
        res = inner_cv(solver, vars; fixed_l2=l2, kw...)
        mse = norm(yval .- Xval * res[1])^2
        
        if mse < best_mse
            best_mse = mse
            best_res = res
            best_idx = curr_idx
            curr_idx += direction
        else
            break # Worsened, stop the line search
        end
    end
    
    best_l2 = lambdas2[best_idx]
    @info "  Best Lambda Pair: (L0 = $(best_res[2]), L2 = $best_l2)"
    
    return best_res
end

function outer_l1_smart_cross_validation(solver, vars; inner_cv=smart_adaptive_cross_validation, kw...)
    X, y, Xval, yval, XTX, p, β⃰, k⃰ = vars
    opt_lasso = find_optimal_lasso(X, y, Xval, yval)
    lambdas1 = 10 .^ range(log10(max(1e-6, 0.1*opt_lasso)), log10(10*opt_lasso), length=15)
    n_grid = length(lambdas1)
    
    # 1. Test Low Extreme
    l1_low = lambdas1[1]
    res_low = inner_cv(solver, vars; fixed_l1=l1_low, kw...)
    mse_low = norm(yval .- Xval * res_low[1])^2
    
    # 2. Test High Extreme
    l1_high = lambdas1[end]
    res_high = inner_cv(solver, vars; fixed_l1=l1_high, kw...)
    mse_high = norm(yval .- Xval * res_high[1])^2
    
    if mse_low < mse_high
        best_res = res_low
        best_mse = mse_low
        best_idx = 1
        direction = 1
    else
        best_res = res_high
        best_mse = mse_high
        best_idx = n_grid
        direction = -1
    end
    
    # 3. Move inwards until MSE worsens
    curr_idx = best_idx + direction
    while 1 <= curr_idx <= n_grid
        l1 = lambdas1[curr_idx]
        res = inner_cv(solver, vars; fixed_l1=l1, kw...)
        mse = norm(yval .- Xval * res[1])^2
        
        if mse < best_mse
            best_mse = mse
            best_res = res
            best_idx = curr_idx
            curr_idx += direction
        else
            break # Worsened, stop the line search
        end
    end
    
    best_l1 = lambdas1[best_idx]
    @info "  Best Lambda Pair: (L0 = $(best_res[2]), L1 = $best_l1)"
    
    return best_res
end

const outer_l1 = outer_l1_smart_cross_validation

# ============================================================================
# Main Experiment
# ============================================================================

function main()
    x⁰ = zeros(Float64, p)
    consecutive_perfect_recoveries = 0
    last_completed_idx = 0

    algo_names = [
        L"NSPG+PGCCD+CPSI1\ (L_0)", 
        L"NSPG+PGCCD+CPSI1\ (L_0L_2)", 
        L"NSPG+CPSI1\ (L_0)", 
        L"NSPG+CPSI1\ (L_0L_2)", 
        L"L0Learn+CPSI1\ (L_0)", 
        L"L0Learn+CPSI1\ (L_0L_2)",
        L"L0Learn+CPSI1\ val\ (L_0)",
        L"L0Learn+CPSI1\ val\ (L_0L_2)"
    ]
    n_algos = length(algo_names)
    @info "Experiment configuration" snr_range trials=T algorithms=algo_names

    Predhist = zeros(length(snr_range), n_algos)
    SUPhist = zeros(length(snr_range), n_algos)
    Infhist = zeros(length(snr_range), n_algos)
    Simhist = zeros(length(snr_range), n_algos)
    Timehist = zeros(length(snr_range), n_algos)

    for (t, snr_val) in enumerate(snr_range)
        @info "SNR=$(snr_val) (n=$n, p=$p)" progress="$(t)/$(length(snr_range))"
        total_start = time()

        for i = 1:T
            vars = variables(corr=corr, ρ=ρ, n=n, p=p, SNR=snr_val, k⃰=k⃰, is_tridiagonal=is_tridiagonal)
            X, y, Xval, yval, XTX, _, β⃰, _ = vars
            
            # Helper for metrics
            suppsim(b) = count(j -> !iszero(β⃰[j]) && !iszero(b[j]), 1:p) / max(k⃰, norm(b, 0))
            predval(b) = norm(X * b .- yval)^2 / norm(yval)^2

            # 1. NSPG+PGCCD+CPSI1 (L0)
            trial_start = time()
            β, best_λ, _, _, _, _, _, _, _ = smart_adaptive_cross_validation((x, f; kw...) -> SolverPSI1(NSPG_PGCCD, x, f; kw...), vars; NSPG=true, use_refinement=true)
            dt = time() - trial_start
            SUPhist[t, 1] += norm(β, 0); Predhist[t, 1] += predval(β); Simhist[t, 1] += suppsim(β); Infhist[t, 1] += norm(β - β⃰, Inf); Timehist[t, 1] += dt
            @show algo_names[1] norm(β, 0) predval(β) suppsim(β) norm(β - β⃰, Inf) round(dt, digits=1)

            # 2. NSPG+PGCCD+CPSI1 (L0L2)
            trial_start = time()
            β, best_λ, _, _, _, _, _, _, _ = outer_l2_smart_cross_validation((x, f; kw...) -> SolverPSI1(NSPG_PGCCD, x, f; kw...), vars; NSPG=true, use_refinement=true)
            dt = time() - trial_start
            SUPhist[t, 2] += norm(β, 0); Predhist[t, 2] += predval(β); Simhist[t, 2] += suppsim(β); Infhist[t, 2] += norm(β - β⃰, Inf); Timehist[t, 2] += dt
            @show algo_names[2] norm(β, 0) predval(β) suppsim(β) norm(β - β⃰, Inf) round(dt, digits=1)

            # 3. NSPG+CPSI1 (L0)
            trial_start = time()
            β, best_λ, _, _, _, _, _, _, _ = smart_adaptive_cross_validation((x, f; kw...) -> SolverPSI1(NSPG, x, f; kw...), vars; NSPG=true, use_refinement=true)
            dt = time() - trial_start
            SUPhist[t, 3] += norm(β, 0); Predhist[t, 3] += predval(β); Simhist[t, 3] += suppsim(β); Infhist[t, 3] += norm(β - β⃰, Inf); Timehist[t, 3] += dt
            @show algo_names[3] norm(β, 0) predval(β) suppsim(β) norm(β - β⃰, Inf) round(dt, digits=1)

            # 4. NSPG+CPSI1 (L0L2)
            trial_start = time()
            β, best_λ, _, _, _, _, _, _, _ = outer_l2_smart_cross_validation((x, f; kw...) -> SolverPSI1(NSPG, x, f; kw...), vars; NSPG=true, use_refinement=true, inner_cv=smart_adaptive_cross_validation)
            dt = time() - trial_start
            SUPhist[t, 4] += norm(β, 0); Predhist[t, 4] += predval(β); Simhist[t, 4] += suppsim(β); Infhist[t, 4] += norm(β - β⃰, Inf); Timehist[t, 4] += dt
            @show algo_names[4] norm(β, 0) predval(β) suppsim(β) norm(β - β⃰, Inf) round(dt, digits=1)

            if L0LEARN_AVAILABLE
                initialize_l0learn_R(vars)

                # 5. L0Learn+CPSI1 (L0) — Inverse CV [COLD]
                trial_start = time()
                β, best_λ, _, _, _, _, _, _, _ = smart_adaptive_cross_validation((x, f; kw...) -> SolverPSI1((args...; k...) -> L0LearnStep(args...; k..., warmstart=false), x, f; kw...), vars; NSPG=false, use_refinement=true)
                dt = time() - trial_start
                SUPhist[t, 5] += norm(β, 0); Predhist[t, 5] += predval(β); Simhist[t, 5] += suppsim(β); Infhist[t, 5] += norm(β - β⃰, Inf); Timehist[t, 5] += dt
                @show algo_names[5] norm(β, 0) predval(β) suppsim(β) norm(β - β⃰, Inf) round(dt, digits=1)

                # 6. L0Learn+CPSI1 (L0L2) — Inverse CV [COLD]
                trial_start = time()
                β, best_λ, _, _, _, _, _, _, _ = outer_l2_smart_cross_validation((x, f; kw...) -> SolverPSI1((args...; k...) -> L0LearnStep(args...; k..., warmstart=false), x, f; kw...), vars; NSPG=false, use_refinement=true, inner_cv=smart_adaptive_cross_validation)
                dt = time() - trial_start
                SUPhist[t, 6] += norm(β, 0); Predhist[t, 6] += predval(β); Simhist[t, 6] += suppsim(β); Infhist[t, 6] += norm(β - β⃰, Inf); Timehist[t, 6] += dt
                @show algo_names[6] norm(β, 0) predval(β) suppsim(β) norm(β - β⃰, Inf) round(dt, digits=1)

                # 7. L0Learn Native (L0)
                trial_start = time()
                β = pure_l0learn_solver_R(vars; maxSuppSize=p, l2_val=0.0)
                dt = time() - trial_start
                SUPhist[t, 7] += norm(β, 0); Predhist[t, 7] += predval(β); Simhist[t, 7] += suppsim(β); Infhist[t, 7] += norm(β - β⃰, Inf); Timehist[t, 7] += dt
                @show algo_names[7] norm(β, 0) predval(β) suppsim(β) norm(β - β⃰, Inf) round(dt, digits=1)

                # 8. L0Learn Native (L0L2)
                trial_start = time()
                β = pure_l0learn_solver_R(vars; maxSuppSize=p, l2_val=0.1) # Use a fixed L2 as in your typical scenario
                dt = time() - trial_start
                SUPhist[t, 8] += norm(β, 0); Predhist[t, 8] += predval(β); Simhist[t, 8] += suppsim(β); Infhist[t, 8] += norm(β - β⃰, Inf); Timehist[t, 8] += dt
                @show algo_names[8] norm(β, 0) predval(β) suppsim(β) norm(β - β⃰, Inf) round(dt, digits=1)
            end

            @info "Trial $i/$T completed"
        end

        @info "Completed SNR=$(snr_val) in $(round(time() - total_start, digits=1))s"
        last_completed_idx = t
        
        # Check early stopping (we stop if all average similarities are >= 0.999)
        avg_sims = Simhist[t, :] ./ T
        if all(x -> x >= 0.999, avg_sims)
            consecutive_perfect_recoveries += 1
            if consecutive_perfect_recoveries >= 2
                @info "Perfect recovery achieved for 2 consecutive SNR steps. Stopping early."
                break
            end
        else
            consecutive_perfect_recoveries = 0
        end
    end

    @info "All experiments finished"

    # Trim and Normalize
    snr_final = collect(snr_range)[1:last_completed_idx]
    
    SUPhist = SUPhist[1:last_completed_idx, :] ./ T
    Predhist = Predhist[1:last_completed_idx, :] ./ T
    Simhist = Simhist[1:last_completed_idx, :] ./ T
    Infhist = Infhist[1:last_completed_idx, :] ./ T
    Timehist = Timehist[1:last_completed_idx, :] ./ T

    # Display results
    println("\n" * "="^60)
    println("RESULTS")
    println("="^60)

    names = reshape(algo_names, 1, :)
    plotname = "SNR_sweep_$(corr)-$(ρ)-n$(n)-p$(p)-k$(k⃰)-T$(T)_$(snr_range_str)"
    specifics = is_tridiagonal ? "_tridiagonal" : "_standard"
    
    theme(:seaborn_bright)
    default(lw=3)

    println("\nGenerating plots...")

    # For SNR, X axis should be logarithmic
    pPred = plot(snr_final, Predhist, labels=names, xscale=:log10, xlabel="SNR", ylabel=L"\frac{\left\Vert Ax-b\right\Vert^2}{\left\Vert b\right\Vert^2}", left_margin=15mm, dpi=600)
    savefig(pPred, "snrPred-$(plotname)$(specifics).png")

    pSim = plot(snr_final, Simhist, labels=names, xscale=:log10, xlabel="SNR", ylabel=L"\frac{|S\cap S^\dagger|}{\max\{|S|,k^\dagger\}}", left_margin=15mm, dpi=600)
    savefig(pSim, "snrSim-$(plotname)$(specifics).png")

    pTime = plot(snr_final, Timehist, labels=names, xscale=:log10, xlabel="SNR", ylabel="Time (s)", left_margin=15mm, dpi=600, legend=:topleft)
    savefig(pTime, "snrTime-$(plotname)$(specifics).png")

    pInf = plot(snr_final, Infhist, labels=names, xscale=:log10, xlabel="SNR", ylabel=L"\left\Vert x-x^\dagger\right\Vert_\infty", left_margin=15mm, dpi=600)
    savefig(pInf, "snrInf-$(plotname)$(specifics).png")

    println("Plots saved.")
end


# Redefine pure_l0learn solvers to handle 8-value vars unpacking from snr_experiment.jl
function pure_l0learn_solver(vars; maxSuppSize=size(vars[1], 2))
    X, y, Xval, yval, XTX, p, β⃰, k⃰ = vars
    RCall.globalEnv[:X_train] = X
    RCall.globalEnv[:y_train] = y
    RCall.globalEnv[:X_val] = Xval
    RCall.globalEnv[:y_val] = yval
    RCall.globalEnv[:maxSuppSize] = maxSuppSize

    try
        r_code = raw"""
        if (dir.exists("~/R_libs")) .libPaths(c("~/R_libs", .libPaths()))
        library(L0Learn)
        fit <- L0Learn.fit(X_train, y_train, penalty="L0", algorithm="CDPSI", maxSuppSize=maxSuppSize, intercept=FALSE)
        best_mse <- Inf
        best_coef <- NULL
        best_lambda <- 0
        lambdas <- fit$lambda[[1]]
        for (i in 1:length(lambdas)) {
            beta <- as.numeric(coef(fit, lambda=lambdas[i], gamma=0))
            mse <- mean((y_val - as.numeric(X_val %*% beta))^2)
            if (mse < best_mse) {
                best_mse <- mse
                best_coef <- beta
                best_lambda <- lambdas[i]
            }
        }
        list(beta=best_coef, lambda=best_lambda)
        """
        res = RCall.rcopy(RCall.reval(RCall.rparse(r_code)))
        β = res[:beta]
        λ = res[:lambda]
        
        suppsim(b) = count(i -> !iszero(β⃰[i]) && !iszero(b[i]), 1:p) / max(k⃰, norm(b, 0))
        predval(b) = norm(X * b .- yval)^2 / norm(yval)^2
        
        return β, λ, norm(β, 0), predval(β), suppsim(β), norm(β - β⃰, Inf)
    catch e
        @warn "Pure L0Learn solver failed: $e"
        return zeros(p), 0.0, 0, 1.0, 0.0, 1.0
    end
end

function pure_l0learn_cd_solver(vars; maxSuppSize=size(vars[1], 2))
    X, y, Xval, yval, XTX, p, β⃰, k⃰ = vars
    RCall.globalEnv[:X_train] = X
    RCall.globalEnv[:y_train] = y
    RCall.globalEnv[:X_val] = Xval
    RCall.globalEnv[:y_val] = yval
    RCall.globalEnv[:maxSuppSize] = maxSuppSize

    try
        r_code = raw"""
        if (dir.exists("~/R_libs")) .libPaths(c("~/R_libs", .libPaths()))
        library(L0Learn)
        fit <- L0Learn.fit(X_train, y_train, penalty="L0", algorithm="CD", maxSuppSize=maxSuppSize, intercept=FALSE)
        best_mse <- Inf
        best_coef <- NULL
        best_lambda <- 0
        lambdas <- fit$lambda[[1]]
        for (i in 1:length(lambdas)) {
            beta <- as.numeric(coef(fit, lambda=lambdas[i], gamma=0))
            mse <- mean((y_val - as.numeric(X_val %*% beta))^2)
            if (mse < best_mse) {
                best_mse <- mse
                best_coef <- beta
                best_lambda <- lambdas[i]
            }
        }
        list(beta=best_coef, lambda=best_lambda)
        """
        res = RCall.rcopy(RCall.reval(RCall.rparse(r_code)))
        β = res[:beta]
        λ = res[:lambda]
        
        suppsim(b) = count(i -> !iszero(β⃰[i]) && !iszero(b[i]), 1:p) / max(k⃰, norm(b, 0))
        predval(b) = norm(X * b .- yval)^2 / norm(yval)^2
        
        return β, λ, norm(β, 0), predval(β), suppsim(β), norm(β - β⃰, Inf)
    catch e
        @warn "Pure L0Learn CD solver failed: $e"
        return zeros(p), 0.0, 0, 1.0, 0.0, 1.0
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end