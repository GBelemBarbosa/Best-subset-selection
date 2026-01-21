#!/usr/bin/env julia
# Debug script to trace smart_adaptive_cross_validation and inverse_cross_validation behavior
# CV functions are EXACT copies from new_test_terminal_cdss_comparison.jl (minus log prints)

using LinearAlgebra
using Distributions
using Random
using ThreadsX

Random.seed!(123)

# ============================================================================
# Parameters - using exp correlation (the problematic case)
# ============================================================================
const p = 1000
const n = 200
const k⃰ = 20
const SNR = 5
const ρ = 0.9
const corr = "exp"
const kₘₐₓ = 1000
const ϵ = 10^-7

println("="^60)
println("DEBUG: Adaptive CV High Support Issue")
println("="^60)
println("Parameters: p=$p, n=$n, k⃰=$k⃰, SNR=$SNR, ρ=$ρ, corr=$corr")
println()

# ============================================================================
# Variable Generation (simplified from main script)
# ============================================================================
function generate_vars()
    Σ = corr == "exp" ? [ρ^abs(i - j) for i = 1:p, j = 1:p] : [1 - (1 - ρ) * (i != j) for i = 1:p, j = 1:p]
    d = MvNormal(zeros(p), Σ)
    X = rand(d, n)'
    for i = 1:p
        X[:, i] ./= norm(view(X, :, i))
    end
    β⃰ = zeros(p)
    βsign = rand([-1, 1], k⃰)
    for i = 1:k⃰
        β⃰[i] = βsign[i]
    end
    σ = norm(X * β⃰) / (sqrt(n) * SNR)
    y = X * β⃰ + σ * randn(n)
    yval = X * β⃰ + σ * randn(n)
    XTX = X' * X
    return X, y, yval, XTX, p, β⃰, k⃰
end

# ============================================================================
# funcs helper
# ============================================================================
function funcs(X, y, yval, XTX, p, β⃰, k⃰, λ)
    HT = sqrt(2 * λ)
    r!(r, β) = (mul!(r, X, β); r .-= y)
    r(β) = X * β - y
    f(r) = norm(r)^2 / 2
    h(β) = λ * norm(β, 0)
    F(r, β) = f(r) + h(β)
    ∇f!(∇f, r) = mul!(∇f, X', r)
    ∇f(r) = X'r
    proxl0(x) = (abs(x) >= HT) * x
    proxl0(x, τ) = (abs.(x) .>= sqrt(2λ * τ)) .* x

    # In-place versions for reduced allocations
    function proxl0!(out, x, τ)
        thresh = sqrt(2λ * τ)
        @inbounds @simd for i in eachindex(out)
            out[i] = abs(x[i]) >= thresh ? x[i] : zero(eltype(x))
        end
        return out
    end
    return r!, r, ∇f!, f, F, ∇f, proxl0, proxl0, proxl0!, proxl0!, X, XTX
end

# ============================================================================
# CDSS (simplified) - WITH DEBUG PRINTS
# ============================================================================
function CDSS(x⁰, funcs_tuple; sortperc=1 / 4, ActiveSetNum=10, verbose=false, kwargs...)
    r!, r, ∇f!, f, F, ∇f, proxl0, _, proxl0!, _, X, XTX = funcs_tuple
    xᵏ = copy(x⁰)
    rᵏ = -r(xᵏ)
    Fxᵏ⁻¹ = F(rᵏ, xᵏ)
    n_vars = length(x⁰)
    greedy = sortperm([abs(dot(view(X, :, i), rᵏ)) for i = 1:n_vars], rev=true)
    Order = greedy[1:Int(ceil(sortperc * n_vars))]
    xᵏ⁻¹ = zeros(length(x⁰))
    Stabilized = false
    SameSuppCounter = 0

    if verbose
        println("      [CDSS] Start: F=$(round(Fxᵏ⁻¹, sigdigits=4)), SUP=$(Int(norm(x⁰, 0))), max|β|=$(round(maximum(abs.(x⁰)), sigdigits=4))")
    end

    for k = 1:kₘₐₓ
        if !Stabilized
            if k > 1 && all(iszero(xᵏ[i]) == iszero(xᵏ⁻¹[i]) for i in eachindex(xᵏ))
                SameSuppCounter += 1
                if SameSuppCounter >= ActiveSetNum
                    Stabilized = true
                    Order = [i for i = 1:n_vars if !iszero(xᵏ[i])]
                end
            else
                SameSuppCounter = 0
            end
        end
        copyto!(xᵏ⁻¹, xᵏ)
        max_before = maximum(abs.(xᵏ))
        @inbounds for i in Order
            xi = proxl0(dot(rᵏ, view(X, :, i)) + xᵏ[i])
            if xi != xᵏ[i]
                BLAS.axpy!(xᵏ[i] - xi, view(X, :, i), rᵏ)
                xᵏ[i] = xi
            end
        end
        max_after = maximum(abs.(xᵏ))
        Fxᵏ = F(rᵏ, xᵏ)
        conv_ratio = (Fxᵏ⁻¹ - Fxᵏ) / Fxᵏ

        if verbose && k <= 5  # Only print first 5 iterations
            println("      [CDSS] k=$k: F=$(round(Fxᵏ, sigdigits=4)), SUP=$(Int(norm(xᵏ, 0))), max|β| $(round(max_before, sigdigits=4))→$(round(max_after, sigdigits=4)), conv=$(round(conv_ratio, sigdigits=4))")
        end

        if conv_ratio <= ϵ
            if verbose
                println("      [CDSS] Converged at k=$k (conv=$conv_ratio <= ϵ=$ϵ)")
            end
            return xᵏ, k
        end
        Fxᵏ⁻¹ = Fxᵏ
    end
    if verbose
        println("      [CDSS] Max iter reached ($kₘₐₓ)")
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

# SolverPSI1 simplified (just use CDSS directly for debugging)
function debug_solver(x⁰, funcs_tuple; γₖ=0.0, verbose=false, kwargs...)
    β, k = SPG(x⁰, funcs_tuple; verbose=verbose)
    return β, k, 1
end

# ============================================================================
# inverse_cross_validation - EXACT COPY from comparison script
# ============================================================================
function debug_inverse_cv(solver, vars; λ_max_ratio=1e30, SPG=false, stagnation_handling=true)
    X, y, yval, XTX, p, β⃰, k⃰ = vars
    suppsim(β) = count(i -> !iszero(β⃰[i]) && !iszero(β[i]), 1:p) / max(k⃰, norm(β, 0))
    predval(β) = norm(X * β .- yval)^2 / norm(yval)^2

    β = β_best = zeros(p)
    best = best_λ = Inf

    # Compute initial residual correlations (residual = y when β = 0)
    ∇fxᵏ = -X'y
    correlations = abs.(∇fxᵏ)

    # Initial λ: use max_corr for inverse CV (we start low and go high)
    # Using min_corr fails for exp correlation where min_corr ≈ 0
    min_corr = ThreadsX.minimum(correlations)

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
    λ_explosion_limit = 1e30

    println("\n--- INVERSE CV DEBUG ---")
    println("Initial: min_corr=$min_corr, λ_initial=$λ")

    while λ < λ_max
        # Ensure λ stays positive
        λ = max(λ, eps())

        # Run solver
        β, kᵢ, kₒ = solver(β, funcs(X, y, yval, XTX, p, β⃰, k⃰, λ); γₖ=SPG ? γₖ : 0.0)

        mse = norm(yval .- X * β)
        if mse < best
            β_best = copy(β)
            best = mse
            best_λ = λ
        end

        SUP = Int(norm(β, 0))
        print("\rIter $i | λ=$(round(λ, sigdigits=6)) | SUP=$SUP | MSE=$(round(mse, sigdigits=4)) | Best λ=$(round(best_λ, sigdigits=4))")
        flush(stdout)

        # Stop if support is zero
        if norm(β, 0) == 0
            println("  -> SUP=0, breaking")
            break
        end

        if SPG
            ∇fxᵏ = X' * (X * β - y)
            XTX∇fxᵏ = X' * X * ∇fxᵏ
            γₖ = dot(∇fxᵏ, ∇fxᵏ) / dot(XTX∇fxᵏ, ∇fxᵏ)
        else
            γₖ = 1.0
        end

        # Compute new λ from formula - ensure monotonic increase for inverse CV
        computed_λ = (0.9^-1) * ThreadsX.minimum(abs(β[j] - γₖ * dot(view(X, :, j), X * β - y)) for j = 1:p if !iszero(β[j]))^2 / (2 * γₖ)

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

        if λ > λ_explosion_limit
            println("  -> λ explosion, breaking")
            break
        end

        i += 1
    end

    println("\n--- FINAL RESULT ---")
    println("Best λ=$best_λ, SUP=$(Int(norm(β_best, 0))), MSE=$best")

    return β_best, best_λ, norm(β_best, 0), predval(β_best), suppsim(β_best)
end

# ============================================================================
# smart_adaptive_cross_validation - EXACT COPY from comparison script
# ============================================================================
function debug_adaptive_cv(solver, vars;
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

    println("\n--- ADAPTIVE CV DEBUG ---")
    println("Initial correlations: max=$max_corr, min=$min_corr")
    println("λ_high=$λ_high, λ_low=$λ_low")

    # Probe Low
    β_low, _, _ = solver(zeros(p), funcs(X, y, yval, XTX, p, β⃰, k⃰, λ_low); γₖ=SPG ? γₖ : 0.0)
    mse_low = calc_mse(β_low)

    # Probe High
    β_high, _, _ = solver(zeros(p), funcs(X, y, yval, XTX, p, β⃰, k⃰, λ_high); γₖ=SPG ? γₖ : 0.0)
    mse_high = calc_mse(β_high)

    println("β_high: SUP=$(Int(norm(β_high, 0))), MSE=$mse_high")
    println("β_low: SUP=$(Int(norm(β_low, 0))), MSE=$mse_low")

    # Select Best Start
    if mse_low < mse_high
        current_λ = λ_low
        current_β = β_low
        current_mse = mse_low
        direction = :increase
        λ_limit = λ_high * λ_max_ratio
        println("Starting direction: INCREASE (low MSE is better)")
    else
        current_λ = λ_high
        current_β = β_high
        current_mse = mse_high
        direction = :decrease
        λ_limit = λ_low * λ_min_ratio
        println("Starting direction: DECREASE (high MSE is better)")
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
        computed_λ = NaN

        if direction == :increase
            SUP = Int(norm(current_β, 0))
            rᵏ = X * current_β - y
            val = SUP == 0 ? 0.0 : ThreadsX.minimum(abs(current_β[j] - γₖ * dot(view(X, :, j), rᵏ)) for j = 1:p if !iszero(current_β[j]))
            computed_λ = val^2 / (2 * γₖ)
            computed_λ = (0.9^-1) * max(computed_λ, current_λ)
        else # :decrease
            SUP = Int(norm(current_β, 0))
            rᵏ = X * current_β - y
            val = SUP == p ? 0.0 : ThreadsX.maximum(abs(dot(view(X, :, j), rᵏ)) for j = 1:p if iszero(current_β[j]))
            computed_λ = γₖ * val^2 / 2
            # CRITICAL: For decrease direction, computed_λ MUST be smaller than current_λ
            computed_λ = 0.9 * min(computed_λ, current_λ)
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
        probe_β, _, _ = solver(current_β, funcs(X, y, yval, XTX, p, β⃰, k⃰, probe_λ); γₖ=SPG ? γₖ : 0.0, verbose=true)
        probe_mse = calc_mse(probe_β)

        probe_max_β = maximum(abs.(probe_β))
        println("\n  Iter $i | λ=$(round(probe_λ, sigdigits=4)) | SUP=$(Int(norm(probe_β, 0))) | max|β|=$(round(probe_max_β, sigdigits=4)) | MSE=$(round(probe_mse, sigdigits=4))")

        # C. Check Improvement (require 1% improvement to stay in current direction)
        min_improvement = 0.01
        if probe_mse >= current_mse
            # Try switching direction
            alt_direction = (direction == :increase) ? :decrease : :increase
            println("[NO IMPROVE] probe_mse=$probe_mse >= current_mse=$current_mse, trying alt_direction=$alt_direction")

            # Propose Alt Lambda
            alt_λ = NaN
            if alt_direction == :increase
                SUP = Int(norm(current_β, 0))
                rᵏ = X * current_β - y
                val = SUP == 0 ? 0.0 : ThreadsX.minimum(abs(current_β[j] - γₖ * dot(view(X, :, j), rᵏ)) for j = 1:p if !iszero(current_β[j]))
                alt_λ = (0.9^-1) * val^2 / (2 * γₖ)
                println("  [ALT INCREASE] val=$val, alt_λ=$alt_λ")
            else # :decrease
                SUP = Int(norm(current_β, 0))
                rᵏ = X * current_β - y
                val = SUP == p ? 0.0 : ThreadsX.maximum(abs(dot(view(X, :, j), rᵏ)) for j = 1:p if iszero(current_β[j]))
                alt_λ = 0.9 * γₖ * val^2 / 2
                println("  [ALT DECREASE] val=$val, alt_λ=$alt_λ")
            end

            alt_β, _, _ = solver(current_β, funcs(X, y, yval, XTX, p, β⃰, k⃰, alt_λ); γₖ=SPG ? γₖ : 0.0)
            alt_mse = calc_mse(alt_β)
            alt_SUP = Int(norm(alt_β, 0))
            alt_max_β = maximum(abs.(alt_β))
            println("  [ALT RESULT] alt_λ=$alt_λ, alt_SUP=$alt_SUP, max|β|=$(round(alt_max_β, sigdigits=4)), alt_mse=$alt_mse")

            # Pick whichever direction gives the smaller MSE
            if alt_mse < probe_mse * (1 - min_improvement)
                # Alt direction is better
                println("  [SWITCH] alt_mse=$alt_mse < probe_mse=$probe_mse, switching to $alt_direction")
                probe_λ = alt_λ
                probe_β = alt_β
                probe_mse = alt_mse
                direction = alt_direction

                # Oscillation check
                direction_flip_count += 1
                if direction_flip_count >= 3
                    println("  [STOP] Oscillation detected (local minima), stopping.")
                    break
                end
                # Update λ_limit for new direction
                λ_limit = (direction == :increase) ? λ_high * λ_max_ratio : λ_low * λ_min_ratio

                # Reset stagnation on switch
                prev_computed_λ = probe_λ
            else
                println("  [NO SWITCH] alt_mse=$alt_mse >= probe_mse=$probe_mse, keeping $direction")
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

    println("\n--- FINAL RESULT ---")
    println("Best λ=$best_λ, SUP=$(Int(norm(best_β, 0))), MSE=$best_mse")

    return best_β, best_λ, norm(best_β, 0), predval(best_β), suppsim(best_β)
end

# ============================================================================
# RUN DEBUG
# ============================================================================
println("\nGenerating test data...")
vars = generate_vars()

println("\nRunning debug_adaptive_cv (SPG=true)...")
β_adapt, λ_adapt, sup_adapt, pred_adapt, sim_adapt = debug_adaptive_cv(debug_solver, vars; SPG=true)

println("\n" * "="^60)
println("\nRunning debug_inverse_cv (SPG=true)...")
β_inv, λ_inv, sup_inv, pred_inv, sim_inv = debug_inverse_cv(debug_solver, vars; SPG=true)

println("\n" * "="^60)
println("SUMMARY")
println("="^60)
println("Adaptive CV: SUP=$(Int(sup_adapt)), λ=$λ_adapt, Pred=$pred_adapt, Sim=$sim_adapt")
println("Inverse CV: SUP=$(Int(sup_inv)), λ=$λ_inv, Pred=$pred_inv, Sim=$sim_inv")
