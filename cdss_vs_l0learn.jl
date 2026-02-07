# CDSS vs L0Learn+PSI1 Hybrid Comparison
# Tests L0Learn's coordinate descent with PSI1 verification using CDSS CV framework
# Based on new_test_terminal.jl experimental setup

@info "Loading packages..."
using Random
using Distributions
using LinearAlgebra
using BenchmarkTools, Profile, TimerOutputs
ENV["GKSwstype"] = "100"
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
    @warn "Continuing without L0Learn comparison..."
end

@info "Packages loaded successfully"

# ============================================================================
# Parameters (can be overridden via command-line arguments)
# Usage: julia script.jl [corr] [ρ] [p] [SNR] [k⃰] [ns_start:ns_step:ns_end] [T]
# Example: julia script.jl exp 0.9 1000 5 20 100:100:1000 10
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
T = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 10

# Parse tridiagonal flag
is_tridiagonal = length(ARGS) >= 9 ? parse(Bool, ARGS[9]) : false

@info "Parameters" corr ρ p SNR k⃰ ns_range T L0LEARN_AVAILABLE is_tridiagonal
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

function funcs(X, y, XTX, λ₀)
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
        @inbounds for i in eachindex(out)
            out[i] = abs(x[i]) >= sqrt(2λ₀ * Uₖ[i]) ? x[i] : zero(eltype(x))
        end
        return out
    end

    return r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, y, XTX
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
        XTX∇fxᵏ = XTX * ∇fxᵏ
        γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
    end
    nsᵏ = γₖ
    lastₘ = fill(Fxᵏ, m)

    for k = 1:kₘₐₓ
        Fxₗ₍ₖ₎ = maximum(lastₘ)

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

    # Active set tracking
    xᵏ⁻¹ = copy(xᵏ)
    SameSuppCounter = 0
    Stabilized = false
    Order = greedy

    for k = 1:kₘₐₓ
        if !Stabilized
            if has_same_support(xᵏ, xᵏ⁻¹)
                SameSuppCounter += 1
                if SameSuppCounter == ActiveSetNum - 1
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
# PSI1 Algorithm
# ============================================================================

function PSI1(xˡ, funcs)
    r!, r, ∇f!, f, F, ∇f, proxl0, proxl0VM, proxl0!, proxl0VM!, X, y, XTX = funcs

    r⃰ = -X'r(xˡ)

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

    seen_supports = Set{UInt64}()
    consecutive_fails = 0

    while !isPSI1 && kₒ < kₘₐₓ
        kₒ += 1
        β, k = solver(β, funcs; γₖ=γₖ, kwargs...)
        kᵢ += k
        γₖ = 0.0
        β, isPSI1 = PSI1(β, funcs)

        if !isPSI1
            consecutive_fails += 1
            support_hash = hash(findall(!iszero, β))
            if support_hash in seen_supports
                break
            end
            push!(seen_supports, support_hash)

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
# Cross Validation - STANDARD (with tracking of which refinement wins)
# ============================================================================

function cross_validation_with_tracking(solver, vars; λ_min_ratio=10^-5, SPG=false, stagnation_handling=true)
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

    λ = 1.01 * γₖ * ThreadsX.maximum(abs(∇fxᵏ[j]) for j = 1:p)^2 / 2
    λ_min = λ * λ_min_ratio

    i = 1
    stagnant_count = 0
    prev_computed_λ = λ

    while λ > λ_min && norm(β, 0) != p
        # Pass lambda_val=λ so solvers that need it (like L0LearnStep) can use it
        β, kᵢ, kₒ = solver(β, funcs(X, y, XTX, λ); γₖ=SPG ? γₖ : 0.0, lambda_val=λ, X_data=X, y_data=y)

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
            ∇fxᵏ = X' * (X * β - y)
            XTX∇fxᵏ = X' * X * ∇fxᵏ
            γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
        end

        # Fast and robust lambda computation
        r_tmp = X * β - y
        grad_tmp = X' * r_tmp
        max_abs_g = 0.0
        @inbounds for j in 1:p
            if iszero(β[j])
                av = abs(grad_tmp[j])
                if av > max_abs_g
                    max_abs_g = av
                end
            end
        end
        raw_λ = γₖ * max_abs_g^2 / 2
        computed_λ = 0.9 * min(prev_computed_λ, raw_λ)

        if stagnation_handling
            if abs(computed_λ - prev_computed_λ) / max(prev_computed_λ, eps()) < 0.01
                stagnant_count += 1
                new_λ = computed_λ > λ ? λ : computed_λ
                if stagnant_count >= 3
                    new_λ = 0.9 * λ
                    stagnant_count = 0
                end
            else
                stagnant_count = 0
                new_λ = computed_λ
            end
            prev_computed_λ = computed_λ
            λ = new_λ
        else
            λ = computed_λ
        end
        i += 1
    end

    # Final refinement with BOTH options and TRACKING
    β_best_refined, kᵢ, kₒ = solver(β_best, funcs(X, y, XTX, best_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=best_λ, X_data=X, y_data=y)
    β_from_zero, kᵢ, kₒ = solver(zeros(p), funcs(X, y, XTX, best_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=best_λ, X_data=X, y_data=y)
    
    pred_refined = predval(β_best_refined)
    pred_zero = predval(β_from_zero)
    chose_beta_best = pred_refined <= pred_zero
    
    β_final = chose_beta_best ? β_best_refined : β_from_zero

    return β_final, best_λ, norm(β_final, 0), predval(β_final), suppsim(β_final), norm(β_final - β⃰, Inf), chose_beta_best
end

# ============================================================================
# Cross Validation - BETA BEST ONLY (no zero start comparison)
# ============================================================================

function cross_validation_beta_best_only(solver, vars; λ_min_ratio=10^-5, SPG=false, stagnation_handling=true)
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

    λ = 1.01 * γₖ * ThreadsX.maximum(abs(∇fxᵏ[j]) for j = 1:p)^2 / 2
    λ_min = λ * λ_min_ratio

    i = 1
    stagnant_count = 0
    prev_computed_λ = λ

    while λ > λ_min && norm(β, 0) != p
        β, kᵢ, kₒ = solver(β, funcs(X, y, XTX, λ); γₖ=SPG ? γₖ : 0.0, lambda_val=λ, X_data=X, y_data=y)

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
            ∇fxᵏ = X' * (X * β - y)
            XTX∇fxᵏ = X' * X * ∇fxᵏ
            γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
        end

        # Fast and robust lambda computation
        r_tmp = X * β - y
        grad_tmp = X' * r_tmp
        max_abs_g = 0.0
        @inbounds for j in 1:p
            if iszero(β[j])
                av = abs(grad_tmp[j])
                if av > max_abs_g
                    max_abs_g = av
                end
            end
        end
        raw_λ = γₖ * max_abs_g^2 / 2
        computed_λ = 0.9 * min(prev_computed_λ, raw_λ)

        if stagnation_handling
            if abs(computed_λ - prev_computed_λ) / max(prev_computed_λ, eps()) < 0.01
                stagnant_count += 1
                new_λ = computed_λ > λ ? λ : computed_λ
                if stagnant_count >= 3
                    new_λ = 0.9 * λ
                    stagnant_count = 0
                end
            else
                stagnant_count = 0
                new_λ = computed_λ
            end
            prev_computed_λ = computed_λ
            λ = new_λ
        else
            λ = computed_λ
        end
        i += 1
    end

    # Final refinement - ONLY β_best, track change
    β_before = copy(β_best)
    β_best, kᵢ, kₒ = solver(β_best, funcs(X, y, XTX, best_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=best_λ, X_data=X, y_data=y)
    
    # Track if refinement changed β significantly
    norm_before = norm(β_before)
    refinement_change = norm_before > eps() ? norm(β_best - β_before) / norm_before : norm(β_best - β_before)

    return β_best, best_λ, norm(β_best, 0), predval(β_best), suppsim(β_best), norm(β_best - β⃰, Inf), refinement_change
end

# ============================================================================
# L0Learn Wrapper (using RCall)
# ============================================================================

function run_l0learn_validation(X_train, y_train, X_val, y_val; maxSuppSize=50)
    if !L0LEARN_AVAILABLE
        return nothing, Inf, 0
    end
    
    n_train, p_dim = size(X_train)
    
    # Use RCall functions directly instead of macros
    try
        RCall.reval(RCall.rparse(".libPaths(unique(c('/home/gbelem/R/library', .libPaths())))"))
        RCall.reval(RCall.rparse("suppressMessages(library(L0Learn))"))
        
        RCall.globalEnv[:X_train] = collect(X_train)
        RCall.globalEnv[:y_train] = collect(y_train)
        RCall.globalEnv[:X_val] = collect(X_val)
        RCall.globalEnv[:y_val] = collect(y_val)
        RCall.globalEnv[:maxSuppSize] = maxSuppSize
        
        r_code = raw"""
        fit <- L0Learn.fit(X_train, y_train, penalty="L0", maxSuppSize=maxSuppSize, intercept=FALSE)
        lambdas <- fit$lambda[[1]]
        best_mse <- Inf
        best_coef <- NULL
        
        for (i in 1:length(lambdas)) {
            beta <- as.numeric(coef(fit, lambda=lambdas[i], gamma=0))
            pred <- X_val %*% beta
            mse <- mean((y_val - as.numeric(pred))^2)
            if (mse < best_mse) {
                best_mse <- mse
                best_coef <- beta
            }
        }
        
        result_beta <- best_coef
        result_mse <- best_mse
        result_supp <- sum(abs(best_coef) > 1e-10)
        """
        RCall.reval(RCall.rparse(r_code))
        
        res_beta = RCall.rcopy(RCall.reval(RCall.rparse("result_beta")))
        res_mse = RCall.rcopy(RCall.reval(RCall.rparse("result_mse")))
        res_supp = RCall.rcopy(RCall.reval(RCall.rparse("result_supp")))
        
        return res_beta, res_mse, Int(res_supp)
    catch e
        @warn "L0Learn validation failed: $e"
        p = size(X_train, 2)
        return zeros(p), Inf, 0
    end
end

# Helper: Solve for a single lambda using L0Learn in R
# optimized to avoid re-parsing R code every time
# We use a Ref to store the parsed script pointer lazily
const R_SOLVE_SCRIPT_REF = Ref{Any}(nothing)

function get_r_solve_script()
    if R_SOLVE_SCRIPT_REF[] === nothing
        R_SOLVE_SCRIPT_REF[] = RCall.rparse(raw"""
            # X_train, y_train are assummed to be in globalEnv
            # Ensure custom library is prioritised
            .libPaths(unique(c("/home/gggt4/R/library", .libPaths())))
            suppressMessages(library(L0Learn))
            
            fit <- L0Learn.fit(X_train, y_train, penalty="L0", lambdaGrid=list(c(lambda_val)), 
                               maxSuppSize=maxSuppSize, intercept=FALSE, initialBeta=beta_init)
            
            # Check if we got a fit
            if (length(fit$lambda[[1]]) > 0) {
                beta <- coef(fit, lambda=lambda_val, gamma=0)
                as.numeric(beta)
            } else {
                numeric(ncol(X_train))
            }
        """)
    end
    return R_SOLVE_SCRIPT_REF[]
end

function solve_l0_cv_R(X, y, algorithm="CD"; maxSuppSize=size(X, 2))
    try
        RCall.globalEnv[:X_train] = collect(X)
        RCall.globalEnv[:y_train] = collect(y)
        RCall.reval(raw"""
            .libPaths(unique(c("/home/gbelem/R/library", .libPaths())))
            suppressMessages(library(L0Learn))
        """)
        script = raw"cvfit <- L0Learn.cvfit(X_train, y_train, penalty='L0', algorithm='" * algorithm * raw"', maxSuppSize=" * string(maxSuppSize) * raw", intercept=FALSE); best_idx <- which.min(cvfit$cvMeans[[1]]); as.numeric(cvfit$fit$beta[[1]][, best_idx])"
        return RCall.rcopy(RCall.reval(script))
    catch e
        @warn "R L0Learn CV failed: $e"
        return zeros(size(X, 2))
    end
end

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
            
            # Scale lambda: lambda_R = lambda_Julia / n
            # Julia objective: 1/2 * ||y - Xb||^2 + lambda * ||b||_0
            # L0Learn objective: 1/(2n) * ||y - Xb||^2 + lambda_R * ||b||_0
            n <- nrow(X_train)
            lambda_scaled <- lambda_val / n
            
            fit <- L0Learn.fit(X_train, y_train, penalty="L0", algorithm=algo, lambdaGrid=list(c(lambda_scaled)), 
                               maxSuppSize=maxSuppSize, intercept=FALSE)
            if (length(fit$lambda[[1]]) > 0) {
                b <- as.numeric(coef(fit, lambda=lambda_scaled, gamma=0))
                # cat("R solved lambda:", lambda_scaled, "SUP:", sum(b!=0), "\n")
                b
            } else {
                numeric(ncol(X_train))
            }
        """
        return RCall.rcopy(RCall.reval(script))
    catch e
        @warn "L0Learn R solve failed: $e"
        return zeros(length(x_init))
    end
end


# L0Learn Step Solver
# Conforms to solver(x, funcs; kwargs...) interface
# Ignores 'funcs' for the optimization part (uses R), but respects 'lambda_val'
function L0LearnStep(xᵏ, funcs; lambda_val= nothing, X_data=nothing, y_data=nothing, algorithm="CD", kwargs...)
    if lambda_val === nothing
        return xᵏ, 0, 0
    end
    
    # Use the R implementation for the Step Solver
    X = X_data !== nothing ? X_data : (length(funcs) >= 11 ? funcs[11] : nothing)
    y = y_data !== nothing ? y_data : (length(funcs) >= 12 ? funcs[12] : nothing)

    if X === nothing || y === nothing
        @warn "L0LearnStep: X or y not found"
        return xᵏ, 0, 0
    end

    # Set data in R for this specific solver call (necessary for different CV folds)
    RCall.globalEnv[:X_train] = collect(X)
    RCall.globalEnv[:y_train] = collect(y)

    beta_new = solve_l0_single_lambda_R(lambda_val, xᵏ, X, y; maxSuppSize=size(X, 2), algorithm=algorithm)
    
    return beta_new, 1, 1
end

# Deprecated/Unused legacy function (keeping empty/commented to mostly preserve file structure if needed, 
# or we can simply replace it entirely as we are doing).
# run_l0learn_with_psi1 removed in favor of standard CV loop integration.

# ============================================================================
# Smart Adaptive Cross Validation (from original file)
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
    # Pass lambda_val=λ so solvers that need it can use it
    β_low, _, _ = solver(zeros(p), funcs(X, y, XTX, λ_low); γₖ=SPG ? γₖ : 0.0, lambda_val=λ_low, X_data=X, y_data=y)
    mse_low = calc_mse(β_low)

    β_high, _, _ = solver(zeros(p), funcs(X, y, XTX, λ_high); γₖ=SPG ? γₖ : 0.0, lambda_val=λ_high, X_data=X, y_data=y)
    mse_high = calc_mse(β_high)

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
    direction_flip_count = 0
    i = 2

    while true
        if direction == :increase
            SUP = Int(norm(current_β, 0))
            rᵏ = X * current_β - y
            val = SUP == 0 ? 0.0 : ThreadsX.minimum(abs(current_β[j] - γₖ * dot(view(X, :, j), rᵏ)) for j = 1:p if !iszero(current_β[j]))
            raw_λ = val^2 / (2 * γₖ)
            computed_λ = (0.9^-1) * max(current_λ, raw_λ)
        else
            SUP = Int(norm(current_β, 0))
            rᵏ = X * current_β - y
            val = SUP == p ? 0.0 : ThreadsX.maximum(abs(dot(view(X, :, j), rᵏ)) for j = 1:p if iszero(current_β[j]))
            raw_λ = γₖ * val^2 / 2
            computed_λ = 0.9 * min(current_λ, raw_λ)
        end

        if stagnation_handling
            if abs(computed_λ - prev_computed_λ) / max(prev_computed_λ, eps()) < 0.01
                stagnant_count += 1
                if direction == :increase
                    computed_λ = computed_λ < current_λ ? current_λ : computed_λ
                    if stagnant_count >= 3
                        computed_λ = current_λ * 1.1
                        stagnant_count = 0
                    end
                else
                    computed_λ = computed_λ > current_λ ? current_λ : computed_λ
                    if stagnant_count >= 3
                        computed_λ = current_λ * 0.9
                        stagnant_count = 0
                    end
                end
            else
                stagnant_count = 0
            end
        end
        prev_computed_λ = computed_λ

        if direction == :increase && computed_λ > λ_limit
            break
        end
        if direction == :decrease && computed_λ < λ_limit
            break
        end

        probe_λ = computed_λ
        probe_β, _, _ = solver(current_β, funcs(X, y, XTX, probe_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=probe_λ, X_data=X, y_data=y)
        probe_mse = calc_mse(probe_β)

        min_improvement = 0.01
        if probe_mse >= current_mse
            alt_direction = (direction == :increase) ? :decrease : :increase

            if alt_direction == :increase
                SUP = Int(norm(current_β, 0))
                rᵏ = X * current_β - y
                val = SUP == 0 ? 0.0 : ThreadsX.minimum(abs(current_β[j] - γₖ * dot(view(X, :, j), rᵏ)) for j = 1:p if !iszero(current_β[j]))
                raw_λ = val^2 / (2 * γₖ)
                alt_λ = (0.9^-1) * max(current_λ, raw_λ)
            else
                SUP = Int(norm(current_β, 0))
                rᵏ = X * current_β - y
                val = SUP == p ? 0.0 : ThreadsX.maximum(abs(dot(view(X, :, j), rᵏ)) for j = 1:p if iszero(current_β[j]))
                raw_λ = γₖ * val^2 / 2
                alt_λ = 0.9 * min(current_λ, raw_λ)
            end

            alt_β, _, _ = solver(current_β, funcs(X, y, XTX, alt_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=alt_λ, X_data=X, y_data=y)
            alt_mse = calc_mse(alt_β)

            if alt_mse < probe_mse * (1 - min_improvement)
                probe_λ = alt_λ
                probe_β = alt_β
                probe_mse = alt_mse
                direction = alt_direction

                direction_flip_count += 1
                if direction_flip_count >= 2
                    break
                end
                λ_limit = (direction == :increase) ? λ_high * λ_max_ratio : λ_low * λ_min_ratio
                prev_computed_λ = probe_λ
            end
        end

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

        if SPG
            ∇fxᵏ = X' * (X * current_β - y)
            XTX∇fxᵏ = X' * X * ∇fxᵏ
            γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
        end
        i += 1
    end

    β_best, kᵢ, kₒ = solver(best_β, funcs(X, y, XTX, best_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=best_λ, X_data=X, y_data=y)
    β_best_0, kᵢ, kₒ = solver(zeros(p), funcs(X, y, XTX, best_λ); γₖ=SPG ? γₖ : 0.0, lambda_val=best_λ, X_data=X, y_data=y)
    
    pred_refined = predval(β_best)
    pred_zero = predval(β_best_0)
    chose_beta_best = pred_refined <= pred_zero
    
    best_β = chose_beta_best ? β_best : β_best_0

    return best_β, best_λ, norm(best_β, 0), predval(best_β), suppsim(best_β), norm(best_β - β⃰, Inf), chose_beta_best
end

# ============================================================================
# Main Experiment
# ============================================================================

function main()
    x⁰ = zeros(Float64, p)
    consecutive_perfect_recoveries = 0
    last_completed_idx = 0

    # Algorithm names grouped
    algo_names = [
        "CDSS+PSI1", "CDSS+PSI1 w/o refinement", 
        "SPGpCDSS+PSI1", "SPGpCDSS+PSI1 w/o refinement",
        "L0Learn", "L0Learn+PSI1", "L0Learn+PSI1 val", "L0Learn+PSI1 val w/o refinement"
    ]
    n_algos = length(algo_names)

    # Reordered Labels and specific colors for groups
    labels = reshape(algo_names, 1, :)
    # Blue shades for CDSS, Green for SPG, Purple/Warm for L0Learn, Teal for Hybrid
    colors = [:blue, :skyblue, :green, :lightgreen, :purple, :darkgoldenrod, :deeppink, :teal]

    @info "Experiment configuration" samples = ns_range trials = T algorithms = algo_names

    # Matrix sizes updated to 8
    Predhist = zeros(T, 8)
    SUPhist = zeros(T, 8)
    Infhist = zeros(T, 8)
    Simhist = zeros(T, 8)
    Timehist = zeros(T, 8)
    
    Predhist_mean = zeros(length(ns_range), 8)
    SUPhist_mean = zeros(length(ns_range), 8)
    Infhist_mean = zeros(length(ns_range), 8)
    Simhist_mean = zeros(length(ns_range), 8)
    Timehist_mean = zeros(length(ns_range), 8)

    # Refinement tracking
    beta_best_chosen_count = zeros(length(ns_range))  # For standard CDSS
    refinement_changes = zeros(length(ns_range))       # For β-only variant
    spgpcdss_beta_best_chosen_count = zeros(length(ns_range))  # For SPGpCDSS

    for (t, n) in enumerate(ns_range)
        @info "Sample size n=$n" progress = "$(t)/$(length(ns_range))"
        total_start = time()

        for i = 1:T
            vars = variables(corr=corr, ρ=ρ, n=n, p=p, SNR=SNR, k⃰=k⃰)
            X, y, yval, XTX, _, β⃰_local, _ = vars
            
            # Helper functions for this trial
            suppsim(β) = count(j -> !iszero(β⃰_local[j]) && !iszero(β[j]), 1:length(β)) / max(k⃰, norm(β, 0))
            predval(β) = norm(X * β .- yval)^2 / norm(yval)^2
            
            # Ensure R has the data for this trial
            if L0LEARN_AVAILABLE
                RCall.globalEnv[:X_train] = collect(X)
                RCall.globalEnv[:y_train] = collect(y)
            end

            # --- Group 1: CDSS+PSI1 ---
            # Alg 1: CDSS+PSI1
            trial_start = time()
            β_cdss, best_λ_cdss, SUP, Pred, Sim, Infv, chose_beta_best = cross_validation_with_tracking(
                (x, f; kw...) -> SolverPSI1(CDSS, x, f; kw...), vars)
            elapsed = time() - trial_start
            SUPhist[i, 1] = SUP; Predhist[i, 1] = Pred; Simhist[i, 1] = Sim; Infhist[i, 1] = Infv; Timehist[i, 1] = elapsed
            beta_best_chosen_count[t] += chose_beta_best ? 1 : 0
            @show "CDSS+PSI1" SUP Pred Sim Infv round(elapsed, digits=2)

            # Alg 2: CDSS+PSI1 w/o refinement
            trial_start = time()
            β, best_λ, SUP, Pred, Sim, Infv, refinement_change = cross_validation_beta_best_only(
                (x, f; kw...) -> SolverPSI1(CDSS, x, f; kw...), vars)
            elapsed = time() - trial_start
            SUPhist[i, 2] = SUP; Predhist[i, 2] = Pred; Simhist[i, 2] = Sim; Infhist[i, 2] = Infv; Timehist[i, 2] = elapsed
            refinement_changes[t] += refinement_change
            @show "CDSS+PSI1 w/o refinement" SUP Pred Sim Infv round(elapsed, digits=2)

            # --- Group 2: SPGpCDSS+PSI1 ---
            # Alg 3: SPGpCDSS+PSI1
            trial_start = time()
            β, best_λ, SUP, Pred, Sim, Infv, spgpcdss_chose_beta = smart_adaptive_cross_validation(
                (x, f; kw...) -> SolverPSI1(SPGpCDSS, x, f; kw...), vars, SPG=true)
            elapsed = time() - trial_start
            SUPhist[i, 3] = SUP; Predhist[i, 3] = Pred; Simhist[i, 3] = Sim; Infhist[i, 3] = Infv; Timehist[i, 3] = elapsed
            spgpcdss_beta_best_chosen_count[t] += spgpcdss_chose_beta ? 1 : 0
            @show "SPGpCDSS+PSI1" SUP Pred Sim Infv round(elapsed, digits=2)

            # Alg 4: SPGpCDSS+PSI1 w/o refinement
            trial_start = time()
            β, best_λ, SUP, Pred, Sim, Infv, _ = cross_validation_beta_best_only(
                (x, f; kw...) -> SolverPSI1(SPGpCDSS, x, f; kw...), vars, SPG=true)
            elapsed = time() - trial_start
            SUPhist[i, 4] = SUP; Predhist[i, 4] = Pred; Simhist[i, 4] = Sim; Infhist[i, 4] = Infv; Timehist[i, 4] = elapsed
            @show "SPGpCDSS+PSI1 w/o refinement" SUP Pred Sim Infv round(elapsed, digits=2)

            # --- Group 3: L0Learn (R-Backend) ---
            # Alg 5: L0Learn (Pure, with R Internal CV)
            trial_start = time()
            β = solve_l0_cv_R(X, y, "CD")
            elapsed = time() - trial_start
            SUPhist[i, 5] = norm(β, 0); Predhist[i, 5] = norm(yval .- X * β)^2 / norm(yval)^2; Simhist[i, 5] = suppsim(β); Infhist[i, 5] = norm(β - β⃰_local, Inf); Timehist[i, 5] = elapsed
            @show "L0Learn" norm(β, 0) Predhist[i, 5] Simhist[i, 5] Infhist[i, 5] round(elapsed, digits=2)

            # Alg 6: L0Learn+PSI1 (with R Internal CV)
            trial_start = time()
            β = solve_l0_cv_R(X, y, "CDPSI")
            elapsed = time() - trial_start
            SUPhist[i, 6] = norm(β, 0); Predhist[i, 6] = norm(yval .- X * β)^2 / norm(yval)^2; Simhist[i, 6] = suppsim(β); Infhist[i, 6] = norm(β - β⃰_local, Inf); Timehist[i, 6] = elapsed
            @show "L0Learn+PSI1" norm(β, 0) Predhist[i, 6] Simhist[i, 6] Infhist[i, 6] round(elapsed, digits=2)

            # Alg 7: L0Learn+PSI1 val (was L0Hybrid)
            trial_start = time()
            # 1. CDSS for init (already available from Algorithm 1)
            # 2. R CD with warm start using CDSS's best lambda
            β = solve_l0_single_lambda_R(best_λ_cdss, β_cdss, X, y)
            elapsed = time() - trial_start
            SUPhist[i, 7] = norm(β, 0); Predhist[i, 7] = norm(yval .- X * β)^2 / norm(yval)^2; Simhist[i, 7] = suppsim(β); Infhist[i, 7] = norm(β - β⃰_local, Inf); Timehist[i, 7] = elapsed
            @show "L0Learn+PSI1 val" norm(β, 0) Predhist[i, 7] Simhist[i, 7] Infhist[i, 7] round(elapsed, digits=2)

            # Alg 8: L0Learn+PSI1 val w/o refinement (was Hybrid w/o refinement)
            trial_start = time()
            β, best_λ, SUP, Pred, Sim, Infv, _ = cross_validation_beta_best_only(
                (x, f; kw...) -> L0LearnStep(x, f; algorithm="CD", kw...), vars)
            elapsed = time() - trial_start
            SUPhist[i, 8] = SUP; Predhist[i, 8] = Pred; Simhist[i, 8] = Sim; Infhist[i, 8] = Infv; Timehist[i, 8] = elapsed
            @show "L0Learn+PSI1 val w/o refinement" SUP Pred Sim Infv round(elapsed, digits=2)

            @info "Trial $i/$T completed"
            flush(stdout)
        end

        # Calculate means and store
        Predhist_mean[t, :] = mean(Predhist, dims=1)
        SUPhist_mean[t, :] = mean(SUPhist, dims=1)
        Infhist_mean[t, :] = mean(Infhist, dims=1)
        Simhist_mean[t, :] = mean(Simhist, dims=1)
        Timehist_mean[t, :] = mean(Timehist, dims=1)
        
        @info "Completed n=$n in $(round(time() - total_start, digits=1))s"
        last_completed_idx = t

        avg_sims = Simhist_mean[t, :]
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

    # Trim to completed size
    ns_final = collect(ns_range)[1:last_completed_idx]
    
    # Trim to completed *rows* (samples), keeping all columns (algorithms)
    SUPhist_mean = SUPhist_mean[1:last_completed_idx, :]
    Predhist_mean = Predhist_mean[1:last_completed_idx, :]
    Simhist_mean = Simhist_mean[1:last_completed_idx, :]
    Infhist_mean = Infhist_mean[1:last_completed_idx, :]
    Timehist_mean = Timehist_mean[1:last_completed_idx, :]
    
    # Histories are per trial, not per N, so just don't trim them or trim if stored differently
    # But here we are just saving means, so we good.
    beta_best_chosen_count = beta_best_chosen_count[1:last_completed_idx]
    refinement_changes = refinement_changes[1:last_completed_idx]
    spgpcdss_beta_best_chosen_count = spgpcdss_beta_best_chosen_count[1:last_completed_idx]

    # Average results
    SUPhist ./= T
    Predhist ./= T
    Simhist ./= T
    Infhist ./= T
    Timehist ./= T
    beta_best_chosen_count ./= T  # Now percentage
    refinement_changes ./= T
    spgpcdss_beta_best_chosen_count ./= T  # Now percentage

    # Display results
    println("\n" * "="^60)
    println("RESULTS")
    println("="^60)
    println("\nSupport Size History (SUPhist):")
    display(SUPhist)
    println("\n\nExecution Time History (Timehist) - seconds:")
    display(Timehist)

    # Refinement Statistics
    println("\n" * "="^60)
    println("REFINEMENT STATISTICS")
    println("="^60)
    println("\nCDSS - β_best chosen percentage (vs zero start) per n:")
    for (i, n) in enumerate(ns_final)
        println("  n=$n: $(round(100 * beta_best_chosen_count[i], digits=1))% chose β_best")
    end
    println("Overall CDSS: $(round(100 * mean(beta_best_chosen_count), digits=1))% chose β_best")
    
    println("\nSPGpCDSS - β_best chosen percentage (vs zero start) per n:")
    for (i, n) in enumerate(ns_final)
        println("  n=$n: $(round(100 * spgpcdss_beta_best_chosen_count[i], digits=1))% chose β_best")
    end
    println("Overall SPGpCDSS: $(round(100 * mean(spgpcdss_beta_best_chosen_count), digits=1))% chose β_best")
    
    println("\nRefinement change magnitude (β-only variant) per n:")
    for (i, n) in enumerate(ns_final)
        println("  n=$n: avg relative change = $(round(refinement_changes[i], sigdigits=3))")
    end
    println("\nOverall avg refinement change: $(round(mean(refinement_changes), sigdigits=3))")

    # Save refinement statistics to file
    open("refinement_stats_$(corr)-$(ρ)-$(p).txt", "w") do f
        println(f, "REFINEMENT STATISTICS")
        println(f, "Parameters: corr=$corr, ρ=$ρ, p=$p, SNR=$SNR, k*=$k⃰, T=$T")
        println(f, "="^60)
        println(f, "\nCDSS - β_best chosen percentage (vs zero start):")
        for (i, n) in enumerate(ns_final)
            println(f, "  n=$n: $(round(100 * beta_best_chosen_count[i], digits=1))%")
        end
        println(f, "Overall CDSS: $(round(100 * mean(beta_best_chosen_count), digits=1))%")
        println(f, "\nSPGpCDSS - β_best chosen percentage (vs zero start):")
        for (i, n) in enumerate(ns_final)
            println(f, "  n=$n: $(round(100 * spgpcdss_beta_best_chosen_count[i], digits=1))%")
        end
        println(f, "Overall SPGpCDSS: $(round(100 * mean(spgpcdss_beta_best_chosen_count), digits=1))%")
        println(f, "\nRefinement change magnitude (CDSS w/o refinement variant):")
        for (i, n) in enumerate(ns_final)
            println(f, "  n=$n: $(round(refinement_changes[i], sigdigits=3))")
        end
        println(f, "\nOverall: $(round(mean(refinement_changes), sigdigits=3))")
    end

    # Plot settings
    names = reshape(algo_names, 1, :)
    plotname = "L0Hybrid_bestlast_$(corr)-$(ρ)-$(p)-$(SNR)-$(k⃰)-$(T)-$(first(ns_range))_$(step(ns_range))_$(last(ns_range))"
    
    # Global plot theme settings for "popy" look
    theme(:seaborn_bright)
    default(lw=3)

    # Create and save plots
    println("\nGenerating plots...")
    
    # Use means for all plots
    valid_ns = ns_final[1:last_completed_idx]
    P_mean = Predhist_mean[1:last_completed_idx, :]
    S_mean = SUPhist_mean[1:last_completed_idx, :]
    I_mean = Infhist_mean[1:last_completed_idx, :]
    Sim_mean = Simhist_mean[1:last_completed_idx, :]
    T_mean = Timehist_mean[1:last_completed_idx, :]

    # Reshape colors to 1xN for plotting series
    color_matrix = reshape(colors, 1, :)

    pPred = plot(valid_ns, P_mean, labels=labels, color=color_matrix, xlabel=L"n", 
                 ylabel=L"\frac{\Vert Ax-b\Vert^2}{\Vert b\Vert^2}", 
                 left_margin=5mm, dpi=600, legend=:topright)
    savefig(pPred, "$(plotname)_Pred.png")

    # Legend for Sim: empty space usually bottom-right for this increasing Sim
    pSim = plot(valid_ns, Sim_mean, labels=labels, color=color_matrix, xlabel=L"n", 
                ylabel=L"\frac{|Supp(x)\cap Supp(x^\dagger)|}{\max\{|Supp(x)|,k^\dagger\}}", 
                left_margin=5mm, dpi=600, legend=:bottomright)
    savefig(pSim, "$(plotname)_Sim.png")

    pInf = plot(valid_ns, I_mean, labels=labels, color=color_matrix, xlabel=L"n", 
                ylabel=L"\Vert x-x^\dagger\Vert_\infty", 
                left_margin=5mm, dpi=600, legend=:topright)
    savefig(pInf, "$(plotname)_Inf.png")

    pSUP = plot(valid_ns, S_mean, labels=labels, color=color_matrix, xlabel=L"n", 
                ylabel=L"\Vert x\Vert_0", 
                left_margin=5mm, dpi=600, legend=:topright)
    savefig(pSUP, "$(plotname)_SUP.png")

    pTime = plot(valid_ns, T_mean, labels=labels, color=color_matrix, xlabel=L"n", 
                 ylabel="Execution Time (s)", 
                 left_margin=5mm, dpi=600, legend=:bottomright)
    savefig(pTime, "$(plotname)_Time.png")

    @info "Plots saved" files = [
        "$(plotname)_Pred.png",
        "$(plotname)_Sim.png",
        "$(plotname)_Inf.png",
        "$(plotname)_SUP.png",
        "$(plotname)_Time.png"
    ]

    println("\n" * "="^60)
    println("All experiments completed successfully!")
    println("="^60)

    return SUPhist, Predhist, Simhist, Infhist, Timehist, beta_best_chosen_count, refinement_changes, spgpcdss_beta_best_chosen_count
end

# Run the main function
main()
