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

# Parse T (number of trials)
T = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 10

# Parse tridiagonal flag
is_tridiagonal = length(ARGS) >= 8 ? parse(Bool, ARGS[8]) : false

@info "Parameters" corr ρ p SNR k⃰ ns_range T is_tridiagonal L0LEARN_AVAILABLE
flush(stdout)

kₘₐₓ = 1000
ϵ = 10^-7

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

    return r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX
end

# ============================================================================
# VMSPG Algorithm
# ============================================================================

function VMSPG(x⁰, funcs; m=15, δ=0.01, τ=0.25, γₘᵢₙ=eps(), γₘₐₓ=typemax(Int64), µ=10^-3)
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
        # Initial SPG step via BB1: γₖ = ⟨sᵏ,sᵏ⟩/⟨yᵏ,sᵏ⟩
        # sᵏ = ε∇f, yᵏ = ∇fxᵏ + X'(y - X(β - ε∇f)) = εX'X∇f
        # γₖ = ε‖∇f‖² / ⟨yᵏ, ∇f⟩ = ε‖∇f‖² / (ε⟨X'X∇f,∇f⟩)
        # ε = 10^-5
        XTX∇fxᵏ = XTX * ∇fxᵏ
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
# RCall Integration for L0Learn
# ============================================================================

const R_SOLVE_SCRIPT_REF = Ref{Any}(nothing)

function get_r_solve_script()
    if R_SOLVE_SCRIPT_REF[] === nothing
        R_SOLVE_SCRIPT_REF[] = RCall.rparse(raw"""
            # Ensure library is loaded including local lib if exists
            if (dir.exists("~/R_libs")) {
                .libPaths(c("~/R_libs", .libPaths()))
            }
            suppressMessages(library(L0Learn))
            
            fit <- L0Learn.fit(X_train, y_train, penalty="L0", lambdaGrid=list(c(lambda_val)), maxSuppSize=maxSuppSize, intercept=FALSE)
            
            if (length(fit$lambda[[1]]) > 0) {
                # Get coefficients for the first lambda/gamma
                beta <- coef(fit, lambda=fit$lambda[[1]][1], gamma=0)
                beta_vec <- as.numeric(beta)
            } else {
                beta_vec <- rep(0, ncol(X_train))
            }
            beta_vec
        """)
    end
    return R_SOLVE_SCRIPT_REF[]
end

function solve_l0_single_lambda_R(lambda_val, x_init; maxSuppSize=length(x_init))
    try
        RCall.globalEnv[:lambda_val] = lambda_val
        RCall.globalEnv[:maxSuppSize] = maxSuppSize
        RCall.globalEnv[:beta_init] = x_init
        
        # Run the pre-parsed script
        beta_vec = RCall.rcopy(RCall.reval(get_r_solve_script()))
        return beta_vec
    catch e
        @warn "L0Learn R solve failed: $e"
        return zeros(length(x_init))
    end
end

function L0LearnStep(xᵏ, funcs; lambda_val=nothing, X_data=nothing, y_data=nothing, kwargs...)
    if lambda_val === nothing
        return xᵏ, 0
    end
    
    # Use the Julia implementation instead of RCall
    X = X_data !== nothing ? X_data : funcs[11]
    y = y_data !== nothing ? y_data : nothing

    if y === nothing
        beta_new = solve_l0_single_lambda_R(lambda_val, xᵏ)
    else
        beta_new = l0learn_fit_julia(X, y, lambda_val; beta_init=xᵏ)
    end
    
    return beta_new, 1
end

# Pure L0Learn wrapper for comparison (Full fit)
function pure_l0learn_solver(vars; maxSuppSize=size(vars[1], 2))
    X, y, yval, XTX, p, β⃰, k⃰ = vars
    RCall.globalEnv[:X_train] = X
    RCall.globalEnv[:y_train] = y
    RCall.globalEnv[:X_val] = X # Use same for simple comparison or yval if desired
    RCall.globalEnv[:y_val] = yval
    RCall.globalEnv[:maxSuppSize] = maxSuppSize

    try
        r_code = raw"""
        if (dir.exists("~/R_libs")) .libPaths(c("~/R_libs", .libPaths()))
        library(L0Learn)
        fit <- L0Learn.fit(X_train, y_train, penalty="L0", maxSuppSize=maxSuppSize, intercept=FALSE)
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
    # Internal simple CD implementation (since we removed standalone CDSS)
    function local_cdss(x⁰, funcs; kwargs...)
        r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX = funcs
        x = copy(x⁰)
        r_res = -r(x)
        fx_prev = F(r_res, x)
        for k in 1:1000
            for i in 1:length(x)
                xi = proxl0(dot(r_res, view(X, :, i)) + x[i])
                if xi != x[i]
                    BLAS.axpy!(x[i] - xi, view(X, :, i), r_res)
                    x[i] = xi
                end
            end
            fx = F(r_res, x)
            if (fx_prev - fx) / fx <= 1e-7; break; end
            fx_prev = fx
        end
        return x, 0
    end
    x, k = SPG(x⁰, funcs; γₖ=γₖ, kwargs...)
    x, k2 = local_cdss(x, funcs; kwargs...)
    return x, k + k2
end

# ============================================================================
# PSI1 Algorithm
# ============================================================================

function PSI1(xˡ, funcs)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, XTX = funcs

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
# Solver with PSI1
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

function cross_validation(solver, vars; λ_min_ratio=10^-5, SPG=false, stagnation_handling=true, lambda_val=nothing)
    X, y, yval, XTX, p, β⃰, k⃰ = vars
    suppsim(β) = count(i -> !iszero(β⃰[i]) && !iszero(β[i]), 1:p) / max(k⃰, norm(β, 0))
    predval(β) = norm(X * β .- yval)^2 / norm(yval)^2

    β = β_best = zeros(p)
    best = best_λ = Inf

    ∇fxᵏ = -X'y

    if SPG
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
            β_best = β
            best = mse
            best_λ = λ
        end

        if norm(β, 0) == p
            break
        end

        if SPG
            ∇fxᵏ = X' * (X * β - y)
            XTX∇fxᵏ = X' * X * ∇fxᵏ
            γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
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
    β_best, kᵢ, kₒ = solver(β_best, funcs(X, y, yval, XTX, best_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=best_λ, X_data=X, y_data=y)
    β_best_0, kᵢ, kₒ = solver(zeros(p), funcs(X, y, yval, XTX, best_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=best_λ, X_data=X, y_data=y)
    β_best = predval(β_best) > predval(β_best_0) ? β_best_0 : β_best

    return β_best, best_λ, norm(β_best, 0), predval(β_best), suppsim(β_best), norm(β_best - β⃰, Inf)
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

    # Initial λ: use min_corr for inverse CV (we start low and go high)
    min_corr = ThreadsX.minimum(abs(∇fxᵏ[j]) for j = 1:p)

    if SPG
        XTX∇fxᵏ = X' * X * ∇fxᵏ
        γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
    else
        γₖ = 1.0
    end

    # Initial λ for inverse CV: start SMALL to get high support, then increase
    λ = (1.01^-1) * γₖ * min_corr^2 / 2

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
    β_best, kᵢ, kₒ = solver(β_best, funcs(X, y, yval, XTX, best_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=best_λ, X_data=X, y_data=y)
    β_best_0, kᵢ, kₒ = solver(zeros(p), funcs(X, y, yval, XTX, best_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=best_λ, X_data=X, y_data=y)
    β_best = predval(β_best) > predval(β_best_0) ? β_best_0 : β_best
    return β_best, best_λ, norm(β_best, 0), predval(β_best), suppsim(β_best), norm(β_best - β⃰, Inf)
end

function smart_adaptive_cross_validation(solver, vars;
    λ_min_ratio=10^-5,
    λ_max_ratio=floatmax(),
    SPG=false,
    stagnation_handling=true,
    lambda_val=nothing)

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
    β_best, kᵢ, kₒ = solver(best_β, funcs(X, y, yval, XTX, best_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=best_λ, X_data=X, y_data=y)
    β_best_0, kᵢ, kₒ = solver(zeros(p), funcs(X, y, yval, XTX, best_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=best_λ, X_data=X, y_data=y)
    best_β = predval(β_best) > predval(β_best_0) ? β_best_0 : β_best

    return best_β, best_λ, norm(best_β, 0), predval(best_β), suppsim(best_β), norm(best_β - β⃰, Inf)
end

# ============================================================================
# Main Experiment
# ============================================================================

function main()
    x⁰ = zeros(Float64, p)
    consecutive_perfect_recoveries = 0
    last_completed_idx = 0

    algo_names = ["SPG", "SPGpCDSS", "L0Hybrid", "L0Learn"]
    n_algos = length(algo_names)
    @info "Experiment configuration" samples = ns_range trials = T algorithms = algo_names

    Predhist = zeros(length(ns_range), n_algos)
    SUPhist = zeros(length(ns_range), n_algos)
    Infhist = zeros(length(ns_range), n_algos)
    Simhist = zeros(length(ns_range), n_algos)
    Timehist = zeros(length(ns_range), n_algos)

    for (t, n) in enumerate(ns_range)
        @info "Sample size n=$n" progress = "$(t)/$(length(ns_range))"
        total_start = time()

        for i = 1:T
            vars = variables(corr=corr, ρ=ρ, n=n, p=p, SNR=SNR, k⃰=k⃰, is_tridiagonal=is_tridiagonal)

            # 1. SPG (Standard Cross Validation)
            trial_start = time()
            β, best_λ, SUP, Pred, Sim, Infv = cross_validation((x, f; kw...) -> SolverPSI1(SPG, x, f; kw...), vars; SPG=true)
            dt = time() - trial_start
            SUPhist[t, 1] += SUP; Predhist[t, 1] += Pred; Simhist[t, 1] += Sim; Infhist[t, 1] += Infv; Timehist[t, 1] += dt
            @show "SPG" SUP Pred Sim Infv round(dt, digits=1)

            # 2. SPGpCDSS (Smart Adaptive CV)
            trial_start = time()
            β, best_λ, SUP, Pred, Sim, Infv = smart_adaptive_cross_validation((x, f; kw...) -> SolverPSI1(SPGpCDSS, x, f; kw...), vars, SPG=true)
            dt = time() - trial_start
            SUPhist[t, 2] += SUP; Predhist[t, 2] += Pred; Simhist[t, 2] += Sim; Infhist[t, 2] += Infv; Timehist[t, 2] += dt
            @show "SPGpCDSS" SUP Pred Sim Infv round(dt, digits=1)

            if L0LEARN_AVAILABLE
                # Sync data to R once per trial if needed for L0Hybrid/L0Learn
                RCall.globalEnv[:X_train] = collect(vars[1])
                RCall.globalEnv[:y_train] = collect(vars[2])

                # 3. L0Hybrid (Standard Cross Validation - Best strategy found)
                trial_start = time()
                β, best_λ, SUP, Pred, Sim, Infv = cross_validation((x, f; kw...) -> SolverPSI1(L0LearnStep, x, f; kw...), vars; lambda_val=nothing)
                dt = time() - trial_start
                SUPhist[t, 3] += SUP; Predhist[t, 3] += Pred; Simhist[t, 3] += Sim; Infhist[t, 3] += Infv; Timehist[t, 3] += dt
                @show "L0Hybrid" SUP Pred Sim Infv round(dt, digits=1)

                # 4. Pure L0Learn (Internal CV)
                trial_start = time()
                β, best_λ, SUP, Pred, Sim, Infv = pure_l0learn_solver(vars)
                dt = time() - trial_start
                SUPhist[t, 4] += SUP; Predhist[t, 4] += Pred; Simhist[t, 4] += Sim; Infhist[t, 4] += Infv; Timehist[t, 4] += dt
                @show "L0Learn" SUP Pred Sim Infv round(dt, digits=1)
            end

            @info "Trial $i/$T completed"
        end

        @info "Completed n=$n in $(round(time() - total_start, digits=1))s"
        
        last_completed_idx = t

        # Check for early stopping
        avg_sims = Simhist[t, :] ./ T
        if all(x -> x >= 0.999, avg_sims)
            consecutive_perfect_recoveries += 1
            if consecutive_perfect_recoveries >= 2
                @info "Perfect recovery achieved for 2 consecutive n steps. Stopping early."
                break
            end
        else
            consecutive_perfect_recoveries = 0
        end
    end

    @info "All experiments finished"

    # Trim and Normalize
    ns_final = collect(ns_range)[1:last_completed_idx]
    
    SUPhist = SUPhist[1:last_completed_idx, :] ./ T
    Predhist = Predhist[1:last_completed_idx, :] ./ T
    Simhist = Simhist[1:last_completed_idx, :] ./ T
    Infhist = Infhist[1:last_completed_idx, :] ./ T
    Timehist = Timehist[1:last_completed_idx, :] ./ T

    # Display results
    println("\n" * "="^60)
    println("RESULTS")
    println("="^60)
    println("\nSupport Size History (SUPhist):")
    display(SUPhist)

    names = reshape(algo_names, 1, :)
    plotname = "NewTerminal_$(corr)-$(ρ)-$(p)-$(SNR)-$(k⃰)-$(T)-$(first(ns_range))_$(step(ns_range))_$(last(ns_range))"
    specifics = is_tridiagonal ? "_tridiagonal" : "_standard"
    
    # Global plot theme settings for "popy" look
    theme(:seaborn_bright)
    default(lw=3)

    # Create and save plots
    println("\nGenerating plots...")

    pPred = plot(ns_final, Predhist, labels=names, xlabel=LaTeXString(raw"n"), ylabel=LaTeXString(raw"\|Ax-b\|^2 / \|b\|^2"), left_margin=5mm, dpi=600)
    savefig(pPred, "pnPred-$(plotname)$(specifics).png")

    pSim = plot(ns_final, Simhist, labels=names, xlabel=LaTeXString(raw"n"), ylabel=LaTeXString(raw"|S \cap S^*| / \max(|S|, k^*)"), left_margin=5mm, dpi=600)
    savefig(pSim, "pnSim-$(plotname)$(specifics).png")

    pInf = plot(ns_final, Infhist, labels=names, xlabel=LaTeXString(raw"n"), ylabel=LaTeXString(raw"\|x-x^*\|_\infty"), dpi=600)
    savefig(pInf, "pnInf-$(plotname)$(specifics).png")

    pSUP = plot(ns_final, SUPhist, labels=names, xlabel=LaTeXString(raw"n"), ylabel=LaTeXString(raw"\|x\|_0",), dpi=600)
    savefig(pSUP, "pnSUP-$(plotname)$(specifics).png")

    pTime = plot(ns_final, Timehist, labels=names, xlabel=LaTeXString(raw"n"), ylabel="Execution Time (s)", dpi=600)
    savefig(pTime, "pnTime-$(plotname)$(specifics).png")

    @info "Plots saved" files = ["pnPred-$(plotname)$(specifics).png",
        "pnSim-$(plotname)$(specifics).png",
        "pnInf-$(plotname)$(specifics).png",
        "pnSUP-$(plotname)$(specifics).png",
        "pnTime-$(plotname)$(specifics).png"]

    println("\n" * "="^60)
    println("All experiments completed successfully!")
    println("="^60)

    return SUPhist, Predhist, Simhist, Infhist
end

# Run the main function
main()